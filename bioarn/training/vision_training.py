"""Reusable online vision training utilities for Bio-ARN."""

from __future__ import annotations

import copy
import contextlib
import socket
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import torch

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, MarginGateConfig, RewardConfig, SDMConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.data import CIFAR10Stream, DataSample, StreamingDataSource
from bioarn.preprocessing import PreprocessingPipeline
from bioarn.reward import RewardSystem
from bioarn.scaling import ScaledBioARN
from bioarn.system import BioARNCore


@dataclass
class VisionTrainConfig:
    input_dim: int = 3072
    concept_dim: int = 256
    max_pool_size: int = 1000
    margin_threshold: float = 0.4
    use_batched: bool = True
    batch_size: int = 32
    learning_rate: float = 0.01
    num_train_samples: int = 5000
    num_test_samples: int = 1000
    preprocessing_warmup_samples: int = 200
    curiosity_weight: float = 0.0
    workspace: GNWConfig | None = None


class SyntheticCIFAR10Stream(StreamingDataSource):
    """Deterministic CIFAR-like fallback stream with class structure."""

    image_shape = (3, 32, 32)

    def __init__(
        self,
        num_samples: int,
        *,
        flatten: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        class_labels: Iterable[int] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.num_samples = int(num_samples)
        self.flatten = bool(flatten)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        labels = list(range(10)) if class_labels is None else [int(label) for label in class_labels]
        if not labels:
            raise ValueError("class_labels must be non-empty.")
        self.class_labels = labels
        self._prototypes = self._build_prototypes()
        base_labels = [self.class_labels[index % len(self.class_labels)] for index in range(self.num_samples)]
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(self.num_samples, generator=generator).tolist()
            self._labels = [base_labels[index] for index in order]
        else:
            self._labels = base_labels

    @classmethod
    def _build_prototypes(cls) -> torch.Tensor:
        prototypes = torch.zeros(10, *cls.image_shape, dtype=torch.float32)
        y_coords = torch.linspace(-1.0, 1.0, cls.image_shape[1])
        x_coords = torch.linspace(-1.0, 1.0, cls.image_shape[2])
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = [
            (2, 2),
            (2, 12),
            (2, 22),
            (12, 2),
            (12, 12),
            (12, 22),
            (22, 2),
            (22, 12),
            (22, 22),
            (8, 8),
        ]
        for label, (top, left) in enumerate(positions):
            channel = label % 3
            prototypes[label, channel, top : top + 8, left : left + 8] = 1.0
            secondary_channel = (channel + 1) % 3
            prototypes[label, secondary_channel, top + 2 : top + 6, left + 2 : left + 6] = 0.35
            gradient = (
                (grid_x * (0.05 * ((label % 5) + 1)))
                + (grid_y * (0.04 * ((label // 5) + 1)))
            ).clamp(min=0.0)
            prototypes[label, channel] += gradient
        prototypes += 0.02
        return prototypes.clamp_(0.0, 1.0)

    def __len__(self) -> int:
        return self.num_samples

    def _make_image(self, label: int, generator: torch.Generator) -> torch.Tensor:
        prototype = self._prototypes[int(label)].clone()
        shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
        shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
        prototype = torch.roll(prototype, shifts=(shift_y, shift_x), dims=(1, 2))
        channel_gain = 0.9 + (0.2 * torch.rand(3, 1, 1, generator=generator))
        contrast = 0.9 + (0.2 * torch.rand(1, generator=generator).item())
        noise = torch.randn(prototype.shape, generator=generator) * 0.08
        image = (prototype * channel_gain * contrast) + noise
        return image.clamp_(0.0, 1.0)

    def stream(self) -> Iterator[DataSample]:
        generator = torch.Generator().manual_seed(self.seed)
        for index, label in enumerate(self._labels):
            image = self._make_image(label, generator)
            data = image.reshape(-1) if self.flatten else image
            yield DataSample(
                data=self._move_tensor(data),
                label=int(label),
                modality="vision",
                metadata={"index": index, "split": "synthetic", "dataset": "synthetic-cifar10"},
            )


@contextlib.contextmanager
def _socket_timeout(timeout_seconds: float):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def load_cifar10_or_synthetic(
    *,
    data_dir: str | Path = "data",
    train_samples: int = 6000,
    test_samples: int = 2000,
    seed: int = 0,
    timeout_seconds: float = 20.0,
) -> tuple[StreamingDataSource, StreamingDataSource, str]:
    """Try CIFAR-10 first; fall back to a structured synthetic stream on failure."""

    try:
        with _socket_timeout(timeout_seconds):
            train_stream = CIFAR10Stream(
                split="train",
                data_dir=data_dir,
                flatten=True,
                normalize=True,
                shuffle=True,
                seed=seed,
            )
            test_stream = CIFAR10Stream(
                split="test",
                data_dir=data_dir,
                flatten=True,
                normalize=True,
                shuffle=False,
                seed=seed,
            )
        return train_stream, test_stream, "cifar10"
    except Exception:
        return (
            SyntheticCIFAR10Stream(train_samples, flatten=True, shuffle=True, seed=seed),
            SyntheticCIFAR10Stream(test_samples, flatten=True, shuffle=False, seed=seed + 1),
            "synthetic-cifar10",
        )


class _PrototypeBank:
    def __init__(self) -> None:
        self.prototypes: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}

    def update(self, label: int, concept_direction: torch.Tensor) -> None:
        normalized = normalize(concept_direction.reshape(1, -1)).squeeze(0)
        count = self.counts.get(label, 0) + 1
        if label not in self.prototypes:
            self.prototypes[label] = normalized.detach().clone()
        else:
            updated = ((self.prototypes[label] * self.counts[label]) + normalized) / count
            self.prototypes[label] = normalize(updated.reshape(1, -1)).squeeze(0)
        self.counts[label] = count

    def predict(self, concept_direction: torch.Tensor) -> int | None:
        if not self.prototypes:
            return None
        labels = list(self.prototypes.keys())
        stacked = torch.stack([self.prototypes[label].to(concept_direction) for label in labels], dim=0)
        query = normalize(concept_direction.reshape(1, -1)).expand_as(stacked)
        similarities = cosine_similarity(stacked, query)
        return labels[int(torch.argmax(similarities).item())]

    def clone(self) -> "_PrototypeBank":
        cloned = _PrototypeBank()
        cloned.prototypes = {label: value.clone() for label, value in self.prototypes.items()}
        cloned.counts = dict(self.counts)
        return cloned


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    """Round-robin interleave samples by class label.

    Interleaved presentation (class 0, class 1, …, class 0, class 1, …) helps
    online Hebbian learners build balanced concept representations from the start,
    which reduces the chance of early CCCs over-specialising on whichever class
    happened to appear first in the stream.  Unlabelled samples are appended at
    the end.
    """
    class_buckets: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
        else:
            class_buckets[int(label)].append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int | None]] = []
    classes = sorted(class_buckets.keys())
    while any(class_buckets[c] for c in classes):
        for c in classes:
            if class_buckets[c]:
                interleaved.append(class_buckets[c].pop(0))
    interleaved.extend(unlabeled)
    return interleaved


def _iter_samples(data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]]) -> Iterator[DataSample | tuple[torch.Tensor, int]]:
    if isinstance(data_stream, StreamingDataSource):
        yield from data_stream.stream()
        return
    if hasattr(data_stream, "stream"):
        yield from data_stream.stream()
        return
    yield from data_stream


def _sample_to_tensor_and_label(sample: DataSample | tuple[torch.Tensor, int] | torch.Tensor) -> tuple[torch.Tensor, int | None]:
    if isinstance(sample, DataSample):
        return sample.data.to(torch.float32).reshape(-1), sample.label
    if isinstance(sample, tuple):
        return sample[0].to(torch.float32).reshape(-1), int(sample[1])
    return sample.to(torch.float32).reshape(-1), None


def take_samples(
    data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]],
    num_samples: int,
    *,
    allowed_labels: set[int] | None = None,
) -> list[tuple[torch.Tensor, int | None]]:
    """Collect a bounded list of normalized flat samples from a stream."""

    samples: list[tuple[torch.Tensor, int | None]] = []
    for sample in islice(_iter_samples(data_stream), max(0, int(num_samples * 10))):
        tensor, label = _sample_to_tensor_and_label(sample)
        if allowed_labels is not None and label not in allowed_labels:
            continue
        samples.append((tensor, label))
        if len(samples) >= num_samples:
            break
    return samples


class VisionTrainer:
    """Train Bio-ARN on vision datasets with proper evaluation."""

    def __init__(
        self,
        config: VisionTrainConfig,
        preprocessing: PreprocessingPipeline | None = None,
    ):
        self.config = config
        self.preprocessing = preprocessing
        self.effective_input_dim = self._effective_input_dim()
        self.system = self._build_system(config, input_dim=self.effective_input_dim)
        self.label_bank = _PrototypeBank()
        self.ccc_label_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.ccc_confidence_sums: defaultdict[int, float] = defaultdict(float)
        self.training_history: list[dict[str, object]] = []
        self.last_fired_indices: list[int] = []
        self.curiosity_system = self._build_curiosity_system()

    def _effective_input_dim(self) -> int:
        if self.preprocessing is None:
            return int(self.config.input_dim)
        return int(self.preprocessing.get_output_dim(self.config.input_dim))

    @staticmethod
    def _build_system(config: VisionTrainConfig, *, input_dim: int) -> BioARNCore:
        ccc_features = 64 if input_dim >= 1024 else max(16, min(64, input_dim))
        workspace_config = copy.deepcopy(config.workspace)
        bio_config = BioARNConfig(
            ccc=CCCConfig(
                input_dim=input_dim,
                concept_dim=config.concept_dim,
                num_f1_features=ccc_features,
                f1_top_k=max(8, ccc_features // 4),
                fast_lr=1.0,
                slow_lr=config.learning_rate,
                feedback_lr=config.learning_rate,
                max_pool_size=config.max_pool_size,
            ),
            margin_gate=MarginGateConfig(
                theta_margin=config.margin_threshold,
                theta_margin_lr=0.001,
                theta_resonance=min(0.9, config.margin_threshold + 0.25),
            ),
            sdm=SDMConfig(
                address_dim=max(512, config.concept_dim * 4),
                hamming_radius=max(16, config.concept_dim // 4),
                num_hard_locations=256,
                data_dim=config.concept_dim,
                decay_rate=0.999,
                stdp_window=10,
            ),
            gnw=copy.deepcopy(workspace_config) if workspace_config is not None else GNWConfig(),
            workspace=workspace_config,
            seed=42,
        )
        return ScaledBioARN(bio_config, use_optimized=config.use_batched)

    def _build_curiosity_system(self) -> RewardSystem | None:
        weight = max(0.0, min(float(self.config.curiosity_weight), 1.0))
        if weight <= 0.0:
            return None
        return RewardSystem(
            RewardConfig(
                intrinsic_scale=1.0,
                novelty_threshold=1.15,
                novelty_boost=1.0 + (1.5 * weight),
                novelty_decay=0.9,
                curiosity_weight=weight,
            )
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        prepared = tensor.to(torch.float32).reshape(-1)
        if self.preprocessing is not None:
            prepared = self.preprocessing.transform(prepared).to(torch.float32).reshape(-1)
        return prepared

    def _materialize_samples(
        self,
        data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]],
        target_samples: int,
    ) -> list[tuple[torch.Tensor, int | None]]:
        samples: list[tuple[torch.Tensor, int | None]] = []
        for sample in islice(_iter_samples(data_stream), target_samples):
            samples.append(_sample_to_tensor_and_label(sample))
        return samples

    def _fit_preprocessing(
        self, samples: list[tuple[torch.Tensor, int | None]]
    ) -> tuple[list[tuple[torch.Tensor, int | None]], int]:
        if self.preprocessing is None or self.preprocessing.is_fitted or not samples:
            return samples, 0

        minimum_training_samples = min(64, max(1, len(samples) // 2))
        warmup = min(
            int(self.config.preprocessing_warmup_samples),
            max(0, len(samples) - minimum_training_samples),
        )
        if warmup <= 0:
            return samples, 0

        warmup_batch = torch.stack([tensor for tensor, _ in samples[:warmup]], dim=0)
        self.preprocessing.fit(warmup_batch)
        return samples[warmup:], warmup

    def _recognition_label(
        self,
        concept_direction: torch.Tensor,
        fired_indices: list[int],
    ) -> int | None:
        ccc_votes: defaultdict[int, float] = defaultdict(float)
        for index in fired_indices:
            counts = self.ccc_label_counts.get(index)
            if not counts:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / max(sum(counts.values()), 1)
            ccc_votes[dominant_label] += purity
        if ccc_votes:
            return max(ccc_votes.items(), key=lambda item: item[1])[0]
        return self.label_bank.predict(concept_direction)

    def _pool_stats(self) -> dict[str, float | int]:
        return self.system.ccc_pool.get_pool_stats()

    def _ccc_direction(self, index: int) -> torch.Tensor:
        if hasattr(self.system.ccc_pool, "concept_directions"):
            return self.system.ccc_pool.concept_directions[index].detach().clone()
        return self.system.ccc_pool.cccs[index].concept_direction.detach().clone()

    def _pool_concept(
        self,
        fired_indices: list[int],
        winner_confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not fired_indices:
            return torch.zeros(self.config.concept_dim, dtype=torch.float32), 0.0
        directions = torch.stack([self._ccc_direction(index) for index in fired_indices], dim=0)
        if winner_confidences.numel() == len(fired_indices):
            weights = winner_confidences.to(directions).unsqueeze(-1)
        else:
            weights = torch.ones(len(fired_indices), 1, device=directions.device, dtype=directions.dtype)
        concept = normalize((directions * weights).sum(dim=0, keepdim=True)).squeeze(0)
        return concept, float(weights.squeeze(-1).max().item())

    def _workspace_consensus(
        self,
        fired_indices: list[int],
        winner_confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool]:
        active_cccs: list[tuple[int, torch.Tensor, float]] = []
        for position, index in enumerate(fired_indices):
            confidence = (
                float(winner_confidences[position].item())
                if position < winner_confidences.numel()
                else 0.0
            )
            active_cccs.append((index, self._ccc_direction(index), confidence))

        thought = self.system.stream.think_step(active_cccs, timestep=self.system.timestep)
        self.system.last_thought = thought
        focused_cccs = self.system._workspace_focus_candidates(active_cccs, thought.broadcast)
        vote_result = self.system.fabric.vote(focused_cccs)
        vote_result = self.system._workspace_bias_vote(vote_result, thought.broadcast)
        concept_direction = vote_result.winning_direction.detach().clone()
        return concept_direction, float(vote_result.confidence), vote_result.voter_count == 0

    def _step_pool(self, tensor: torch.Tensor, *, allow_recruit: bool) -> tuple[list[int], torch.Tensor, float, bool]:
        if getattr(self.system.config, "workspace", None) is not None:
            pool = self.system.ccc_pool
            if hasattr(pool, "_vectorized_state") and hasattr(pool, "_ensure_batch"):
                raw_batch, _ = pool._ensure_batch(tensor)
                state = pool._vectorized_state(raw_batch, timestep=self.system.timestep)
                fired_mask = state.fired.any(dim=-1)
                fired_indices = fired_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                winner_confidences = (
                    state.confidence.index_select(0, torch.tensor(fired_indices, device=state.confidence.device)).mean(dim=-1)
                    if fired_indices
                    else torch.empty(0, dtype=torch.float32, device=state.confidence.device)
                )

                if allow_recruit and not state.any_fired:
                    recruit_index, recruit_output = pool.recruit(raw_batch, timestep=self.system.timestep)
                    if recruit_index is not None and recruit_output is not None:
                        fired_indices = [int(recruit_index)]
                        winner_confidences = torch.tensor(
                            [float(recruit_output.confidence.reshape(-1).mean().item())],
                            dtype=torch.float32,
                            device=state.confidence.device,
                        )
            else:
                pool_output = self.system._run_pool(tensor, allow_recruit=allow_recruit)
                fired_indices = list(pool_output.fired_indices)
                winner_confidences = pool_output.winner_confidences

            concept, confidence, abstained = self._workspace_consensus(
                fired_indices,
                winner_confidences.to(torch.float32),
            )
            self.system.timestep += 1
            self.last_fired_indices = fired_indices
            return fired_indices, concept, float(confidence), bool(abstained)

        pool = self.system.ccc_pool
        if hasattr(pool, "_vectorized_state") and hasattr(pool, "_ensure_batch"):
            raw_batch, _ = pool._ensure_batch(tensor)
            state = pool._vectorized_state(raw_batch, timestep=self.system.timestep)
            fired_mask = state.fired.any(dim=-1)
            fired_indices = fired_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            winner_confidences = (
                state.confidence.index_select(0, torch.tensor(fired_indices, device=state.confidence.device)).mean(dim=-1)
                if fired_indices
                else torch.empty(0, dtype=torch.float32, device=state.confidence.device)
            )

            if allow_recruit and not state.any_fired:
                recruit_index, recruit_output = pool.recruit(raw_batch, timestep=self.system.timestep)
                if recruit_index is not None and recruit_output is not None:
                    fired_indices = [int(recruit_index)]
                    winner_confidences = torch.tensor(
                        [float(recruit_output.confidence.reshape(-1).mean().item())],
                        dtype=torch.float32,
                        device=state.confidence.device,
                    )

            self.system.timestep += 1
            concept, confidence = self._pool_concept(fired_indices, winner_confidences)
            self.last_fired_indices = fired_indices
            return fired_indices, concept, confidence, len(fired_indices) == 0

        pool_output = self.system._run_pool(tensor, allow_recruit=allow_recruit)
        self.system.timestep += 1
        fired_indices = list(pool_output.fired_indices)
        concept, confidence = self._pool_concept(fired_indices, pool_output.winner_confidences)
        self.last_fired_indices = fired_indices
        return fired_indices, concept, confidence, len(fired_indices) == 0

    def _record_ccc_activity(
        self,
        fired_indices: list[int],
        label: int | None,
        confidence: float,
    ) -> None:
        if label is None:
            return
        for index in fired_indices:
            self.ccc_label_counts[index][int(label)] += 1
            self.ccc_confidence_sums[index] += float(confidence)

    @staticmethod
    def _estimate_prediction_error(
        *,
        prediction: int | None,
        label: int | None,
        confidence: float,
        abstained: bool,
        recruited: bool,
    ) -> float:
        error = max(0.0, 1.0 - float(confidence))
        if label is None:
            return error + (0.2 if recruited else 0.0)
        if abstained:
            error += 0.7
        elif prediction != label:
            error += 0.5
        else:
            error *= 0.4
        if recruited:
            error += 0.35
        return float(error)

    def _curiosity_replay_count(
        self,
        *,
        reward_step,
        prediction: int | None,
        label: int | None,
        abstained: bool,
        recruited: bool,
        replay_depth: int,
    ) -> int:
        if self.curiosity_system is None or replay_depth > 0:
            return 0
        priority = 0.0
        if bool(reward_step.novelty.is_novel):
            priority += 1.0
        if label is not None and (abstained or prediction != label):
            priority += 1.0
        if recruited:
            priority += 0.5
        if priority * float(self.config.curiosity_weight) < 0.75:
            return 0
        return 1

    def _train_single_sample(
        self,
        tensor: torch.Tensor,
        label: int | None,
    ) -> tuple[int | None, bool, bool, float]:
        before_committed = int(self._pool_stats()["num_committed"])
        tensor = self._prepare_tensor(tensor)
        fired_indices, concept, confidence, abstained_flag = self._step_pool(
            tensor,
            allow_recruit=True,
        )
        prediction = None if abstained_flag else self._recognition_label(concept, fired_indices)
        if not abstained_flag and label is not None:
            self.label_bank.update(int(label), concept)
        self._record_ccc_activity(fired_indices, label, confidence)
        recruited = int(self._pool_stats()["num_committed"]) > before_committed
        return prediction, bool(abstained_flag), recruited, float(confidence)

    @torch.no_grad()
    def train_online(
        self,
        data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]],
        num_samples: int | None = None,
        *,
        num_passes: int = 1,
        interleave_classes: bool = False,
    ) -> dict[str, object]:
        """Train the system online on a stream of samples.

        Args:
            data_stream: Labelled samples to train on.
            num_samples: Cap on samples to materialise from the stream.  Defaults
                to ``config.num_train_samples``.
            num_passes: Number of full passes over the materialised sample list.
                Pass 1 is processed in presentation order; subsequent passes are
                independently shuffled.  Values > 1 improve convergence because
                early samples (trained before any concepts existed) are revisited
                after the prototype bank is populated.
            interleave_classes: When ``True``, reorder the materialised samples
                into round-robin class order before the first pass
                (class 0, class 1, …, class 0, class 1, …).  This prevents long
                runs of the same class from monopolising early CCC slots, which
                helps the system build balanced, well-separated class prototypes —
                particularly useful when the source stream is sorted by class.
        """
        target_samples = self.config.num_train_samples if num_samples is None else int(num_samples)
        num_passes = max(1, int(num_passes))
        samples = self._materialize_samples(data_stream, target_samples)
        train_samples, warmup_samples = self._fit_preprocessing(samples)

        if interleave_classes:
            train_samples = _interleave_by_class(train_samples)

        if self.curiosity_system is not None:
            self.curiosity_system.reset()

        effective_target = len(train_samples) * num_passes
        processed = 0
        correct = 0
        labeled = 0
        abstained = 0
        accuracy_curve: list[float] = []
        raw_accuracy_curve: list[float] = []
        utilization_curve: list[float] = []
        abstention_curve: list[float] = []
        novelty_scores: list[float] = []
        prediction_errors: list[float] = []
        curiosity_drives: list[float] = []
        curiosity_replays = 0
        progress_interval = max(1, max(effective_target, 1) // 10)

        for pass_index in range(num_passes):
            if pass_index == 0:
                pass_samples = train_samples
            else:
                # Shuffle independently for each repeat pass so the system
                # does not memorise a fixed sequence.
                perm = torch.randperm(len(train_samples)).tolist()
                pass_samples = [train_samples[i] for i in perm]

            queue = deque((tensor, label, 0) for tensor, label in pass_samples)
            while queue:
                tensor, label, replay_depth = queue.popleft()
                prediction, abstained_flag, recruited, confidence = self._train_single_sample(tensor, label)

                processed += 1
                if label is not None:
                    labeled += 1
                    correct += int(prediction == label)
                abstained += int(abstained_flag)

                if self.curiosity_system is not None:
                    prediction_error = self._estimate_prediction_error(
                        prediction=prediction,
                        label=label,
                        confidence=confidence,
                        abstained=abstained_flag,
                        recruited=recruited,
                    )
                    reward_step = self.curiosity_system.step(
                        prediction_error,
                        learned=bool(label is not None and prediction == label and not abstained_flag),
                    )
                    prediction_errors.append(prediction_error)
                    novelty_scores.append(float(reward_step.novelty.novelty_score))
                    curiosity_drives.append(float(reward_step.modulation.exploration_drive))
                    replay_count = self._curiosity_replay_count(
                        reward_step=reward_step,
                        prediction=prediction,
                        label=label,
                        abstained=abstained_flag,
                        recruited=recruited,
                        replay_depth=replay_depth,
                    )
                    curiosity_replays += replay_count
                    for _ in range(replay_count):
                        queue.append((tensor.detach().clone(), label, replay_depth + 1))

                pool_stats = self._pool_stats()
                accuracy_curve.append(correct / max(labeled, 1))
                if replay_depth == 0:
                    raw_accuracy_curve.append(correct / max(labeled, 1))
                utilization_curve.append(int(pool_stats["num_committed"]) / self.config.max_pool_size)
                abstention_curve.append(abstained / processed)

                if processed % progress_interval == 0 or processed == effective_target:
                    pass_tag = f"p{pass_index + 1}/{num_passes} " if num_passes > 1 else ""
                    print(
                        f"[train] {pass_tag}{processed}/{effective_target} "
                        f"cccs={int(pool_stats['num_committed'])} "
                        f"acc={accuracy_curve[-1]:.3f} "
                        f"abstain={abstention_curve[-1]:.3f}"
                    )

        result = {
            "processed_samples": processed,
            "raw_samples_seen": len(samples),
            "warmup_samples": warmup_samples,
            "num_passes": num_passes,
            "accuracy": correct / max(labeled, 1),
            "abstention_rate": abstained / max(processed, 1),
            "committed_cccs": int(self._pool_stats()["num_committed"]),
            "accuracy_curve": accuracy_curve,
            "raw_accuracy_curve": raw_accuracy_curve,
            "pool_utilization_curve": utilization_curve,
            "abstention_rate_curve": abstention_curve,
            "curiosity_enabled": self.curiosity_system is not None,
            "curiosity_replays": curiosity_replays,
            "mean_prediction_error": sum(prediction_errors) / max(len(prediction_errors), 1),
            "mean_novelty": sum(novelty_scores) / max(len(novelty_scores), 1),
            "peak_curiosity_drive": max(curiosity_drives, default=0.0),
        }
        if self.curiosity_system is not None:
            result["curiosity_stats"] = self.curiosity_system.get_stats()
        self.training_history.append(result)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        test_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]],
        num_samples: int | None = None,
    ) -> dict[str, object]:
        target_samples = self.config.num_test_samples if num_samples is None else int(num_samples)
        total = 0
        labeled = 0
        correct = 0
        covered = 0
        covered_correct = 0
        abstained = 0
        firing_sum = 0
        per_class_totals: Counter[int] = Counter()
        per_class_correct: Counter[int] = Counter()
        predictions: list[int | None] = []
        labels: list[int | None] = []

        for sample in islice(_iter_samples(test_stream), target_samples):
            tensor, label = _sample_to_tensor_and_label(sample)
            tensor = self._prepare_tensor(tensor)
            fired_indices, concept, _, abstained_flag = self._step_pool(tensor, allow_recruit=False)
            prediction = None if abstained_flag else self._recognition_label(concept, fired_indices)

            total += 1
            labeled += int(label is not None)
            correct += int(label is not None and prediction == label)
            covered += int(prediction is not None)
            covered_correct += int(label is not None and prediction == label and prediction is not None)
            abstained += int(abstained_flag)
            firing_sum += len(fired_indices)
            predictions.append(prediction)
            labels.append(label)
            if label is not None:
                per_class_totals[int(label)] += 1
                per_class_correct[int(label)] += int(prediction == label)

        committed = max(int(self._pool_stats()["num_committed"]), 1)
        per_class_accuracy = {
            label: per_class_correct[label] / max(per_class_totals[label], 1)
            for label in sorted(per_class_totals)
        }
        return {
            "accuracy": correct / max(labeled, 1),
            "covered_accuracy": covered_correct / max(covered, 1),
            "abstention_rate": abstained / max(total, 1),
            "coverage": covered / max(total, 1),
            "per_class_accuracy": per_class_accuracy,
            "pool_utilization": committed / self.config.max_pool_size,
            "mean_firing_count": firing_sum / max(total, 1),
            "mean_firing_fraction": firing_sum / max(total * committed, 1),
            "total_samples": total,
            "predictions": predictions,
            "labels": labels,
        }

    @torch.no_grad()
    def ood_detection_test(
        self,
        noise_samples: torch.Tensor | Iterable[torch.Tensor],
    ) -> dict[str, float]:
        if isinstance(noise_samples, torch.Tensor):
            iterable: Iterable[torch.Tensor] = noise_samples
        else:
            iterable = noise_samples

        total = 0
        abstained = 0
        mean_firing = 0.0
        mean_confidence = 0.0
        ood_threshold = max(self.config.margin_threshold + 0.15, 0.75)
        for sample in iterable:
            fired_indices, _, confidence, abstained_flag = self._step_pool(
                self._prepare_tensor(sample.to(torch.float32).reshape(-1)),
                allow_recruit=False,
            )
            total += 1
            abstained += int(abstained_flag or confidence < ood_threshold)
            mean_firing += len(fired_indices)
            mean_confidence += confidence

        return {
            "abstention_rate": abstained / max(total, 1),
            "mean_firing_count": mean_firing / max(total, 1),
            "mean_confidence": mean_confidence / max(total, 1),
        }

    def continual_learning_test(
        self,
        stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]],
        class_order: list[list[int]],
    ) -> dict[str, object]:
        shadow = VisionTrainer(
            VisionTrainConfig(
                input_dim=self.config.input_dim,
                concept_dim=self.config.concept_dim,
                max_pool_size=self.config.max_pool_size,
                margin_threshold=max(self.config.margin_threshold, 0.55),
                use_batched=self.config.use_batched,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                num_train_samples=self.config.num_train_samples,
                num_test_samples=self.config.num_test_samples,
                preprocessing_warmup_samples=self.config.preprocessing_warmup_samples,
                curiosity_weight=self.config.curiosity_weight,
            ),
            preprocessing=copy.deepcopy(self.preprocessing),
        )
        class_sets = [set(int(label) for label in group) for group in class_order]
        stage_train_limits = max(200, self.config.num_train_samples // max(len(class_sets), 1))
        eval_limit = max(100, self.config.num_test_samples // max(len(class_sets[0]), 1))

        old_classes = class_sets[0]
        old_train = take_samples(stream, stage_train_limits, allowed_labels=old_classes)
        old_eval = take_samples(stream, eval_limit, allowed_labels=old_classes)
        shadow.train_online(old_train, num_samples=len(old_train))
        before = shadow.evaluate(old_eval, num_samples=len(old_eval))

        stage_metrics: list[dict[str, object]] = [before]
        for stage_labels in class_sets[1:]:
            stage_train = take_samples(stream, stage_train_limits, allowed_labels=stage_labels)
            shadow.train_online(stage_train, num_samples=len(stage_train))
            stage_metrics.append(
                shadow.evaluate(
                    take_samples(stream, eval_limit, allowed_labels=old_classes | stage_labels),
                    num_samples=eval_limit,
                )
            )

        after = shadow.evaluate(old_eval, num_samples=len(old_eval))
        forgetting = max(0.0, float(before["accuracy"]) - float(after["accuracy"]))
        retention = float(after["accuracy"]) / max(float(before["accuracy"]), 1e-6)
        return {
            "before_accuracy": float(before["accuracy"]),
            "after_accuracy": float(after["accuracy"]),
            "forgetting": forgetting,
            "retention": retention,
            "stage_metrics": stage_metrics,
            "passed": forgetting < 0.10,
        }

    def get_ccc_analysis(self) -> dict[str, object]:
        pool_stats = self._pool_stats()
        summaries: list[dict[str, object]] = []
        specialized = 0
        purity_sum = 0.0

        for index in range(int(pool_stats["num_committed"])):
            counts = self.ccc_label_counts.get(index, Counter())
            total = sum(counts.values())
            top_classes = counts.most_common(3)
            dominant_label = top_classes[0][0] if top_classes else None
            purity = (top_classes[0][1] / total) if total else 0.0
            specialized += int(purity >= 0.6 and total > 0)
            purity_sum += purity
            summaries.append(
                {
                    "ccc_index": index,
                    "fires": total,
                    "dominant_label": dominant_label,
                    "purity": purity,
                    "top_classes": top_classes,
                    "mean_confidence": self.ccc_confidence_sums[index] / max(total, 1),
                }
            )

        summaries.sort(key=lambda item: (item["purity"], item["fires"]), reverse=True)
        committed = max(int(pool_stats["num_committed"]), 1)
        return {
            "committed_cccs": int(pool_stats["num_committed"]),
            "specialized_cccs": specialized,
            "specialization_rate": specialized / committed,
            "mean_purity": purity_sum / committed,
            "ccc_summaries": summaries,
        }


__all__ = [
    "SyntheticCIFAR10Stream",
    "VisionTrainConfig",
    "VisionTrainer",
    "load_cifar10_or_synthetic",
    "take_samples",
]
