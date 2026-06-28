"""Audio training loop for Bio-ARN synthetic speech-like streams."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Iterator

import torch

from bioarn.config import (
    AudioTrainConfig,
    BioARNConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    SDMConfig,
)
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.data import DataSample, StreamingDataSource, SyntheticAudioStream
from bioarn.hierarchy import AudioHierarchy
from bioarn.preprocessing import AudioPreprocessor
from bioarn.scaling import ScaledBioARN


@dataclass
class _PoolStepResult:
    fired_indices: list[int]
    concept_direction: torch.Tensor
    confidence: float
    abstained: bool
    winner_confidences: torch.Tensor


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


def _iter_samples(
    data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int] | torch.Tensor],
) -> Iterator[DataSample | tuple[torch.Tensor, int] | torch.Tensor]:
    if isinstance(data_stream, StreamingDataSource):
        yield from data_stream.stream()
        return
    if hasattr(data_stream, "stream"):
        yield from data_stream.stream()
        return
    yield from data_stream


def _sample_to_waveform_and_label(
    sample: DataSample | tuple[torch.Tensor, int] | torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    if isinstance(sample, DataSample):
        return sample.data.to(torch.float32).reshape(-1), sample.label
    if isinstance(sample, tuple):
        return sample[0].to(torch.float32).reshape(-1), int(sample[1])
    return sample.to(torch.float32).reshape(-1), None


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    class_buckets: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
        else:
            class_buckets[int(label)].append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int | None]] = []
    classes = sorted(class_buckets.keys())
    while any(class_buckets[label] for label in classes):
        for label in classes:
            if class_buckets[label]:
                interleaved.append(class_buckets[label].pop(0))
    interleaved.extend(unlabeled)
    return interleaved


class AudioTrainer:
    """Online audio classification trainer using mel features and CCC pools."""

    def __init__(self, config: AudioTrainConfig):
        self.config = copy.deepcopy(config)
        self.preprocessor = AudioPreprocessor(self.config.audio)
        self.hierarchy = AudioHierarchy(self.config.hierarchy)
        self.system = self._build_system()
        self.label_bank = _PrototypeBank()
        self.ccc_label_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.training_history: list[dict[str, object]] = []

    def _build_system(self) -> ScaledBioARN:
        belt_dim = int(self.config.hierarchy.belt_dim)
        ccc_features = max(16, min(64, belt_dim))
        margin_config = MarginGateConfig(
            theta_margin=self.config.margin_threshold,
            theta_margin_lr=0.001,
            theta_resonance=min(0.9, self.config.margin_threshold + 0.25),
        )
        bio_config = BioARNConfig(
            ccc=CCCConfig(
                input_dim=belt_dim,
                concept_dim=belt_dim,
                num_f1_features=ccc_features,
                f1_top_k=max(8, ccc_features // 4),
                fast_lr=1.0,
                slow_lr=self.config.learning_rate,
                feedback_lr=self.config.learning_rate,
                max_pool_size=self.config.max_pool_size,
                max_growth_factor=self.config.max_growth_factor,
            ),
            margin_gate=margin_config,
            sdm=SDMConfig(
                address_dim=max(512, belt_dim * 4),
                hamming_radius=max(16, belt_dim // 4),
                num_hard_locations=256,
                data_dim=belt_dim,
                decay_rate=0.999,
                stdp_window=10,
            ),
            gnw=GNWConfig(concept_dim=belt_dim),
            workspace=None,
            device=self.config.device,
            seed=self.config.seed,
        )
        system = ScaledBioARN(bio_config, use_optimized=bool(self.config.use_batched))
        return system.to(torch.device(self.config.device))

    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.preprocessor.waveform_to_mel(waveform.to(self.config.device))
        return self.hierarchy(mel)

    def _materialize_encoded_samples(
        self,
        data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int] | torch.Tensor],
        target_samples: int,
    ) -> list[tuple[torch.Tensor, int | None]]:
        examples: list[tuple[torch.Tensor, int | None]] = []
        for sample in islice(_iter_samples(data_stream), max(1, int(target_samples))):
            waveform, label = _sample_to_waveform_and_label(sample)
            examples.append((self.encode_waveform(waveform).detach().clone(), label))
        return examples

    def _ccc_direction(self, index: int) -> torch.Tensor:
        if hasattr(self.system.ccc_pool, "concept_directions"):
            return self.system.ccc_pool.concept_directions[index].detach().clone()
        return self.system.ccc_pool.cccs[index].concept_direction.detach().clone()

    def _pool_stats(self) -> dict[str, float | int]:
        return self.system.ccc_pool.get_pool_stats()

    def _pool_concept(
        self,
        fired_indices: list[int],
        winner_confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not fired_indices:
            dim = int(self.config.hierarchy.belt_dim)
            return torch.zeros(dim, dtype=torch.float32, device=torch.device(self.config.device)), 0.0
        directions = torch.stack([self._ccc_direction(index).to(self.config.device) for index in fired_indices], dim=0)
        if winner_confidences.numel() == len(fired_indices):
            weights = winner_confidences.to(directions).unsqueeze(-1)
        else:
            weights = torch.ones(len(fired_indices), 1, device=directions.device, dtype=directions.dtype)
        concept = normalize((directions * weights).sum(dim=0, keepdim=True)).squeeze(0)
        return concept, float(weights.squeeze(-1).max().item())

    def _step_pool(self, encoded_audio: torch.Tensor, *, allow_recruit: bool) -> _PoolStepResult:
        pool_output = self.system._run_pool(encoded_audio, allow_recruit=allow_recruit)
        fired_indices = list(pool_output.fired_indices)
        winner_confidences = pool_output.winner_confidences.to(torch.float32)
        concept, confidence = self._pool_concept(fired_indices, winner_confidences)
        self.system.timestep += 1
        return _PoolStepResult(
            fired_indices=fired_indices,
            concept_direction=concept,
            confidence=float(confidence),
            abstained=not fired_indices,
            winner_confidences=winner_confidences.detach().clone(),
        )

    def _recognition_label(self, concept_direction: torch.Tensor, fired_indices: list[int]) -> int | None:
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

    def _record_ccc_activity(self, fired_indices: list[int], label: int | None) -> None:
        if label is None:
            return
        for index in fired_indices:
            self.ccc_label_counts[index][int(label)] += 1

    @torch.no_grad()
    def train_online(
        self,
        stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int] | torch.Tensor] | None = None,
    ) -> dict[str, object]:
        """Train on streaming audio samples and return summary metrics."""

        if stream is None:
            stream = SyntheticAudioStream(
                self.config.num_train_samples,
                sample_rate=self.config.audio.sample_rate,
                duration_ms=self.config.audio.max_duration_ms,
                shuffle=True,
                seed=self.config.seed,
                device=self.config.device,
            )
        examples = self._materialize_encoded_samples(stream, self.config.num_train_samples)
        if self.config.interleave_classes:
            examples = _interleave_by_class(examples)

        processed = 0
        labeled = 0
        correct = 0
        abstained = 0
        accuracy_curve: list[float] = []
        abstention_curve: list[float] = []

        for pass_index in range(self.config.num_passes):
            if pass_index == 0:
                pass_examples = examples
            else:
                order = torch.randperm(
                    len(examples),
                    generator=torch.Generator().manual_seed(self.config.seed + pass_index),
                ).tolist()
                pass_examples = [examples[index] for index in order]

            for encoded_audio, label in pass_examples:
                step = self._step_pool(encoded_audio, allow_recruit=True)
                prediction = None if step.abstained else self._recognition_label(step.concept_direction, step.fired_indices)
                if label is not None:
                    labeled += 1
                    correct += int(prediction == label)
                    self.label_bank.update(int(label), step.concept_direction)
                self._record_ccc_activity(step.fired_indices, label)

                processed += 1
                abstained += int(step.abstained)
                accuracy_curve.append(correct / max(labeled, 1))
                abstention_curve.append(abstained / max(processed, 1))

        result = {
            "processed_samples": processed,
            "num_passes": int(self.config.num_passes),
            "accuracy": correct / max(labeled, 1),
            "abstention_rate": abstained / max(processed, 1),
            "committed_cccs": int(self._pool_stats()["num_committed"]),
            "accuracy_curve": accuracy_curve,
            "abstention_rate_curve": abstention_curve,
        }
        self.training_history.append(result)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int] | torch.Tensor] | None = None,
    ) -> dict[str, object]:
        if stream is None:
            stream = SyntheticAudioStream(
                self.config.num_test_samples,
                sample_rate=self.config.audio.sample_rate,
                duration_ms=self.config.audio.max_duration_ms,
                shuffle=False,
                seed=self.config.seed + 1,
                device=self.config.device,
            )
        examples = self._materialize_encoded_samples(stream, self.config.num_test_samples)

        total = 0
        labeled = 0
        correct = 0
        abstained = 0
        per_class_totals: Counter[int] = Counter()
        per_class_correct: Counter[int] = Counter()

        for encoded_audio, label in examples:
            step = self._step_pool(encoded_audio, allow_recruit=False)
            prediction = None if step.abstained else self._recognition_label(step.concept_direction, step.fired_indices)
            total += 1
            abstained += int(step.abstained)
            if label is not None:
                labeled += 1
                correct += int(prediction == label)
                per_class_totals[int(label)] += 1
                per_class_correct[int(label)] += int(prediction == label)

        per_class_accuracy = {
            label: per_class_correct[label] / max(per_class_totals[label], 1)
            for label in sorted(per_class_totals)
        }
        committed = max(int(self._pool_stats()["num_committed"]), 1)
        capacity = max(int(self._pool_stats()["total_concepts"]), 1)
        return {
            "accuracy": correct / max(labeled, 1),
            "abstention_rate": abstained / max(total, 1),
            "coverage": (total - abstained) / max(total, 1),
            "committed_cccs": committed,
            "pool_utilization": committed / capacity,
            "per_class_accuracy": per_class_accuracy,
            "total_samples": total,
        }


__all__ = ["AudioTrainer"]
