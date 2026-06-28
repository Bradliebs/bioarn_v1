"""Reusable online vision training utilities for Bio-ARN."""

from __future__ import annotations

import copy
import contextlib
import socket
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import torch

from bioarn.config import (
    AugmentationConfig,
    BioARNConfig,
    CCCConfig,
    ConvCCCConfig,
    GNWConfig,
    LateralPredictionConfig,
    MarginGateConfig,
    PrecisionConfig,
    RewardConfig,
    SDMConfig,
)
from bioarn.core.conv_ccc import ConvCCCPool
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.core.margin_gate import ResonanceOutput
from bioarn.data import AugmentedCIFARStream, CIFAR10Stream, DataSample, HebbianAugmentation, StreamingDataSource
from bioarn.loop import SensorimotorLoop
from bioarn.preprocessing import PreprocessingPipeline
from bioarn.reward import NoveltySignal, RewardSystem
from bioarn.scaling import ScaledBioARN
from bioarn.system import BioARNCore
from bioarn.training.curriculum import CurriculumScheduler
from bioarn.training.maturation import MaturationConfig, MaturationSchedule


@dataclass
class VisionTrainConfig:
    input_dim: int = 3072
    concept_dim: int = 256
    max_pool_size: int = 1000
    max_growth_factor: float = 3.0
    margin_threshold: float = 0.4
    use_batched: bool = True
    batch_size: int = 32
    learning_rate: float = 0.01
    consolidation_strength: float = 0.0
    freeze_f1_after: int = 0
    f1_adapter_dim: int = 16
    num_train_samples: int = 5000
    num_test_samples: int = 1000
    preprocessing_warmup_samples: int = 200
    curiosity_weight: float = 0.0
    curriculum: bool = False
    contrastive_curiosity: bool = False
    protection_growth_rate: float = 0.1
    protection_decay_rate: float = 0.01
    replay_interval: int = 64
    enable_elastic_protection: bool = False
    enable_replay: bool = False
    enable_eviction: bool = False
    use_conv_ccc: bool = False
    conv_ccc: ConvCCCConfig | None = None
    pass_lr_decay: float = 1.0
    supervised_merge_threshold: float = 0.2
    workspace: GNWConfig | None = None
    maturation: MaturationConfig | None = None
    precision: PrecisionConfig | None = None
    lateral_prediction: LateralPredictionConfig | None = None
    num_f1_features: int | None = None
    f1_top_k: int | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class PoolStepResult:
    fired_indices: list[int]
    concept_direction: torch.Tensor
    confidence: float
    abstained: bool
    winner_confidences: torch.Tensor
    broadcast_strength: float = 0.0

    def __iter__(self):
        yield self.fired_indices
        yield self.concept_direction
        yield self.confidence
        yield self.abstained


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
    augmentation: AugmentationConfig | None = None,
) -> tuple[StreamingDataSource, StreamingDataSource, str]:
    """Try CIFAR-10 first; fall back to a structured synthetic stream on failure."""

    use_augmentation = augmentation is not None and augmentation.enabled
    augmenter = (
        HebbianAugmentation(
            random_flip=augmentation.random_flip,
            random_crop=augmentation.random_crop,
            color_jitter=augmentation.color_jitter,
            cutout=augmentation.cutout,
            cutout_size=augmentation.cutout_size,
        )
        if use_augmentation
        else None
    )

    try:
        with _socket_timeout(timeout_seconds):
            train_stream: StreamingDataSource = CIFAR10Stream(
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
        if use_augmentation and augmenter is not None:
            train_stream = AugmentedCIFARStream(
                num_samples=train_samples,
                augmentation=augmenter,
                augmentation_factor=augmentation.augmentation_factor,
                seed=seed,
                data_dir=data_dir,
            )
        return train_stream, test_stream, "cifar10"
    except Exception:
        synthetic_train: StreamingDataSource = SyntheticCIFAR10Stream(
            train_samples,
            flatten=True,
            shuffle=True,
            seed=seed,
        )
        if use_augmentation and augmenter is not None:
            synthetic_train = AugmentedCIFARStream(
                num_samples=train_samples,
                augmentation=augmenter,
                augmentation_factor=augmentation.augmentation_factor,
                seed=seed,
                base_stream=SyntheticCIFAR10Stream(
                    train_samples,
                    flatten=False,
                    shuffle=True,
                    seed=seed,
                ),
            )
        return (
            synthetic_train,
            SyntheticCIFAR10Stream(test_samples, flatten=True, shuffle=False, seed=seed + 1),
            "synthetic-cifar10",
        )


class _PrototypeBank:
    def __init__(
        self,
        *,
        max_entries_per_label: int = 8,
        recruit_threshold: float = 0.82,
        vote_top_k: int = 3,
    ) -> None:
        self.max_entries_per_label = int(max(1, max_entries_per_label))
        self.recruit_threshold = float(recruit_threshold)
        self.vote_top_k = int(max(1, vote_top_k))
        self.prototypes: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[int, list[int]] = defaultdict(list)

    def update(self, label: int, concept_direction: torch.Tensor) -> None:
        normalized = normalize(concept_direction.reshape(1, -1)).squeeze(0)
        entries = self.prototypes[int(label)]
        counts = self.counts[int(label)]
        if not entries:
            entries.append(normalized.detach().clone())
            counts.append(1)
            return

        similarities = torch.tensor(
            [float(cosine_similarity(entry, normalized).item()) for entry in entries],
            dtype=torch.float32,
        )
        best_index = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_index].item())
        if best_score < self.recruit_threshold and len(entries) < self.max_entries_per_label:
            entries.append(normalized.detach().clone())
            counts.append(1)
            return

        count = counts[best_index] + 1
        updated = normalize(((entries[best_index] * counts[best_index]) + normalized).reshape(1, -1)).squeeze(0)
        entries[best_index] = updated.detach().clone()
        counts[best_index] = count

    def predict(self, concept_direction: torch.Tensor) -> int | None:
        if not self.prototypes:
            return None
        query = normalize(concept_direction.reshape(1, -1)).squeeze(0)
        label_scores: dict[int, float] = {}
        for label, entries in self.prototypes.items():
            if not entries:
                continue
            stacked = torch.stack([entry.to(query) for entry in entries], dim=0)
            similarities = cosine_similarity(stacked, query.expand_as(stacked))
            top_k = min(self.vote_top_k, similarities.numel())
            top_values, top_indices = torch.topk(similarities, k=top_k)
            score = 0.0
            for similarity, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                weight = float(self.counts[int(label)][int(index)])
                score += max(0.0, float(similarity)) * max(weight, 1.0)
            label_scores[int(label)] = score / max(sum(self.counts[int(label)]), 1)
        if not label_scores:
            return None
        return max(label_scores.items(), key=lambda item: item[1])[0]

    def clone(self) -> "_PrototypeBank":
        cloned = _PrototypeBank(
            max_entries_per_label=self.max_entries_per_label,
            recruit_threshold=self.recruit_threshold,
            vote_top_k=self.vote_top_k,
        )
        cloned.prototypes = defaultdict(
            list,
            {
                label: [value.clone() for value in values]
                for label, values in self.prototypes.items()
            },
        )
        cloned.counts = defaultdict(list, {label: list(values) for label, values in self.counts.items()})
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
        self.config = copy.deepcopy(config)
        self.preprocessing = preprocessing
        self._spatial_input_shape = SensorimotorLoop._infer_visual_shape(int(self.config.input_dim))
        if self.config.use_conv_ccc and self.preprocessing is not None:
            raise ValueError("ConvCCCPool currently expects raw image inputs without preprocessing.")
        self.effective_input_dim = self._effective_input_dim()
        self._configured_workspace = copy.deepcopy(self.config.workspace)
        self.system = self._build_system(self.config, input_dim=self.effective_input_dim)
        self.config.concept_dim = int(getattr(self.system.config.ccc, "concept_dim", self.config.concept_dim))
        if self.config.use_conv_ccc:
            self.config.max_pool_size = int(getattr(self.system.config.ccc, "max_pool_size", self.config.max_pool_size))
        self.label_bank = _PrototypeBank()
        self.ccc_label_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.ccc_confidence_sums: defaultdict[int, float] = defaultdict(float)
        self.training_history: list[dict[str, object]] = []
        self.last_fired_indices: list[int] = []
        self.curiosity_system = self._build_curiosity_system()
        self.curriculum_scheduler = CurriculumScheduler() if self.config.curriculum else None
        self.maturation = self._build_maturation_schedule()
        self._apply_maturation_phase()

    def _effective_input_dim(self) -> int:
        if self.preprocessing is None:
            return int(self.config.input_dim)
        return int(self.preprocessing.get_output_dim(self.config.input_dim))

    def _concept_dim(self) -> int:
        return int(getattr(self.system.config.ccc, "concept_dim", self.config.concept_dim))

    @staticmethod
    def _build_system(config: VisionTrainConfig, *, input_dim: int) -> BioARNCore:
        use_classic_ccc = (
            bool(config.enable_elastic_protection)
            or bool(config.enable_replay)
            or bool(config.enable_eviction)
        )
        margin_config = MarginGateConfig(
            theta_margin=config.margin_threshold,
            theta_margin_lr=0.001,
            theta_resonance=min(0.9, config.margin_threshold + 0.25),
        )
        if config.use_conv_ccc:
            channels, height, width = SensorimotorLoop._infer_visual_shape(input_dim)
            if height != width:
                raise ValueError("ConvCCCPool currently requires square spatial inputs.")
            if config.conv_ccc is not None:
                conv_config = copy.deepcopy(config.conv_ccc)
                conv_config.in_channels = channels
                conv_config.spatial_size = height
                conv_config.hebbian_batch_size = (
                    max(1, int(config.batch_size)) if config.use_batched else 1
                )
            else:
                conv_config = ConvCCCConfig(
                    in_channels=channels,
                    spatial_size=height,
                    num_conv_features=96,
                    num_conv_layers=4,
                    conv_hidden_channels=(48, 96, 128),
                    spatial_grid=4,
                    f1_top_k=96,
                    fast_lr=1.0,
                    slow_lr=config.learning_rate,
                    feedback_lr=config.learning_rate,
                    conv_hebbian_lr=max(0.00075, min(0.006, float(config.learning_rate) * 0.35)),
                    hebbian_batch_size=max(1, int(config.batch_size)) if config.use_batched else 1,
                    conv_competitive_k=24,
                    spatial_top_k=8,
                    conv_weight_norm=1.0,
                    enable_local_contrast_norm=True,
                    contrast_kernel_size=5,
                    response_norm_eps=1e-4,
                    feature_pool_avg_mix=0.35,
                    hebbian_oja_decay=0.08,
                    filter_decorrelation=0.03,
                    max_pool_size=config.max_pool_size,
                    max_growth_factor=config.max_growth_factor,
                    consolidation_strength=config.consolidation_strength,
                    lock_threshold=0.8,
                )
            workspace_config = copy.deepcopy(config.workspace)
            precision_config = copy.deepcopy(config.precision)
            lateral_config = copy.deepcopy(config.lateral_prediction)
            if precision_config is not None:
                precision_config.pool_size = int(conv_config.max_pool_size)
            conv_f1_features = (
                int(config.num_f1_features)
                if config.num_f1_features is not None
                else max(16, min(64, input_dim))
            )
            conv_f1_top_k = (
                int(config.f1_top_k)
                if config.f1_top_k is not None
                else max(8, min(64, conv_f1_features) // 4)
            )
            placeholder_ccc = CCCConfig(
                input_dim=input_dim,
                concept_dim=conv_config.concept_dim,
                num_f1_features=conv_f1_features,
                f1_top_k=conv_f1_top_k,
                freeze_f1_after=config.freeze_f1_after,
                f1_adapter_dim=config.f1_adapter_dim,
                fast_lr=1.0,
                slow_lr=config.learning_rate,
                feedback_lr=config.learning_rate,
                max_pool_size=1,
                max_growth_factor=1.0,
                consolidation_strength=config.consolidation_strength,
                protection_growth_rate=config.protection_growth_rate,
                protection_decay_rate=config.protection_decay_rate,
                replay_interval=config.replay_interval,
                enable_elastic_protection=config.enable_elastic_protection,
                enable_replay=config.enable_replay,
                enable_eviction=config.enable_eviction,
                precision=precision_config,
                lateral_prediction=lateral_config,
            )
            bio_config = BioARNConfig(
                ccc=placeholder_ccc,
                margin_gate=margin_config,
                sdm=SDMConfig(
                    address_dim=max(512, conv_config.concept_dim * 4),
                    hamming_radius=max(16, conv_config.concept_dim // 4),
                    num_hard_locations=256,
                    data_dim=conv_config.concept_dim,
                    decay_rate=0.999,
                    stdp_window=10,
                ),
                gnw=copy.deepcopy(workspace_config) if workspace_config is not None else GNWConfig(),
                workspace=workspace_config,
                seed=42,
            )
            system = ScaledBioARN(bio_config, use_optimized=False)
            system.ccc_pool = ConvCCCPool(conv_config, margin_config)
            system.config.ccc = replace(
                system.config.ccc,
                concept_dim=conv_config.concept_dim,
                max_pool_size=conv_config.max_pool_size,
                max_growth_factor=conv_config.max_growth_factor,
                slow_lr=conv_config.slow_lr,
                feedback_lr=conv_config.feedback_lr,
            )
            system.fabric.ccc_config = system.config.ccc
            return system

        ccc_features = (
            int(config.num_f1_features)
            if config.num_f1_features is not None
            else (64 if input_dim >= 1024 else max(16, min(64, input_dim)))
        )
        ccc_top_k = (
            int(config.f1_top_k)
            if config.f1_top_k is not None
            else max(8, ccc_features // 4)
        )
        workspace_config = copy.deepcopy(config.workspace)
        precision_config = copy.deepcopy(config.precision)
        lateral_config = copy.deepcopy(config.lateral_prediction)
        if precision_config is not None:
            precision_config.pool_size = int(config.max_pool_size)
        bio_config = BioARNConfig(
            ccc=CCCConfig(
                input_dim=input_dim,
                concept_dim=config.concept_dim,
                num_f1_features=ccc_features,
                f1_top_k=ccc_top_k,
                freeze_f1_after=config.freeze_f1_after,
                f1_adapter_dim=config.f1_adapter_dim,
                fast_lr=1.0,
                slow_lr=config.learning_rate,
                feedback_lr=config.learning_rate,
                max_pool_size=config.max_pool_size,
                max_growth_factor=config.max_growth_factor,
                consolidation_strength=config.consolidation_strength,
                protection_growth_rate=config.protection_growth_rate,
                protection_decay_rate=config.protection_decay_rate,
                replay_interval=config.replay_interval,
                enable_elastic_protection=config.enable_elastic_protection,
                enable_replay=config.enable_replay,
                enable_eviction=config.enable_eviction,
                precision=precision_config,
                lateral_prediction=lateral_config,
            ),
            margin_gate=margin_config,
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
        return ScaledBioARN(
            bio_config,
            use_optimized=bool(config.use_batched) and not use_classic_ccc,
        )

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

    def _build_maturation_schedule(self) -> MaturationSchedule | None:
        if self.config.maturation is None or not bool(self.config.maturation.enabled):
            return None
        return MaturationSchedule(copy.deepcopy(self.config.maturation))

    def _apply_maturation_phase(self) -> None:
        if self.maturation is None:
            return
        active_modules = self.maturation.get_active_modules()
        workspace_active = bool(active_modules["workspace"] and self._configured_workspace is not None)
        self.system.config.workspace = (
            copy.deepcopy(self._configured_workspace) if workspace_active else None
        )
        self.system.workspace_enabled = workspace_active

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.use_conv_ccc:
            prepared = tensor.to(torch.float32)
            channels, height, width = self._spatial_input_shape
            expected = channels * height * width
            if prepared.dim() == 1:
                if prepared.numel() != expected:
                    raise ValueError(
                        f"Expected flattened image with {expected} values, got {prepared.numel()}."
                    )
                return prepared.reshape(channels, height, width)
            if prepared.dim() == 3:
                if tuple(prepared.shape) != (channels, height, width):
                    raise ValueError(
                        f"Expected image with shape {(channels, height, width)}, got {tuple(prepared.shape)}."
                    )
                return prepared
            raise ValueError("ConvCCCPool expects per-sample vision inputs to be flat or CHW tensors.")

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
        if self.config.use_conv_ccc and self.preprocessing is None and samples:
            minimum_training_samples = min(64, max(1, len(samples) // 2))
            warmup = min(
                int(self.config.preprocessing_warmup_samples),
                max(0, len(samples) - minimum_training_samples),
            )
            if warmup <= 0:
                return samples, 0
            self._warmup_conv_pool(samples[:warmup])
            return samples, warmup
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

    def _observe_pool_samples(self, sample_count: int) -> None:
        observer = getattr(self.system.ccc_pool, "observe_samples", None)
        if callable(observer):
            observer(int(sample_count))

    def _freeze_pool_f1_if_ready(self) -> None:
        self._flush_pool_hebbian_updates()
        freezer = getattr(self.system.ccc_pool, "freeze_f1", None)
        pool_config = getattr(self.system.config, "ccc", None)
        if not callable(freezer) or pool_config is None:
            return
        if int(getattr(pool_config, "freeze_f1_after", 0)) <= 0:
            return
        freezer()

    def _flush_pool_hebbian_updates(self) -> None:
        flusher = getattr(self.system.ccc_pool, "flush_hebbian_updates", None)
        if callable(flusher):
            flusher()

    def _warmup_conv_pool(self, samples: list[tuple[torch.Tensor, int | None]]) -> None:
        pool = getattr(self.system, "ccc_pool", None)
        shared_f1 = getattr(pool, "shared_f1", None)
        if shared_f1 is None or not samples:
            return

        batch_size = max(
            1,
            int(getattr(getattr(pool, "config", None), "hebbian_batch_size", max(1, self.config.batch_size))),
        )
        warmup_scale = 1.25
        for start in range(0, len(samples), batch_size):
            batch_tensors = [self._prepare_tensor(tensor) for tensor, _ in samples[start : start + batch_size]]
            batch = torch.stack(batch_tensors, dim=0)
            shared_f1.hebbian_update(
                batch,
                learning_signal=torch.ones(batch.shape[0], dtype=torch.float32, device=batch.device),
                lr=float(getattr(pool.config, "conv_hebbian_lr", self.config.learning_rate * 0.25)) * warmup_scale,
            )
        flusher = getattr(pool, "flush_hebbian_updates", None)
        if callable(flusher):
            flusher()

    def start_new_task(self) -> None:
        if int(getattr(self.system.config.ccc, "freeze_f1_after", 0)) <= 0:
            return
        self._flush_pool_hebbian_updates()
        self._freeze_pool_f1_if_ready()
        creator = getattr(self.system.ccc_pool, "create_task_adapter", None)
        if callable(creator):
            creator()

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
            mean_confidence = self.ccc_confidence_sums[index] / max(sum(counts.values()), 1)
            ccc_votes[dominant_label] += purity * max(mean_confidence, 0.25)
        bank_label = self.label_bank.predict(concept_direction)
        if bank_label is not None:
            ccc_votes[int(bank_label)] += 0.35
        if ccc_votes:
            return max(ccc_votes.items(), key=lambda item: item[1])[0]
        return bank_label

    def _pool_stats(self) -> dict[str, float | int]:
        return self.system.ccc_pool.get_pool_stats()

    def _current_pool_capacity(self) -> int:
        return int(self._pool_stats()["total_concepts"])

    def _current_pool_precision(self) -> float:
        getter = getattr(self.system.ccc_pool, "get_precision", None)
        if not callable(getter):
            return 1.0
        return float(getter())

    def _ensure_pool_growth(self, pool) -> object:
        first_uncommitted = getattr(pool, "_first_uncommitted_index", None)
        grow = getattr(pool, "grow", None)
        if not callable(first_uncommitted) or not callable(grow):
            return pool
        if first_uncommitted() is not None:
            return pool
        grown_pool = grow()
        if grown_pool is pool or grown_pool is None:
            return pool
        self.system.ccc_pool = grown_pool
        self.system.config.ccc = grown_pool.config
        return grown_pool

    def _update_pool_importance(
        self,
        fired_indices: list[int],
        winner_confidences: torch.Tensor | None = None,
    ) -> None:
        update_importance = getattr(self.system.ccc_pool, "update_importance", None)
        if callable(update_importance):
            update_importance(fired_indices, confidences=winner_confidences)

    def _update_lateral_predictions(self, fired_indices: list[int]) -> None:
        updater = getattr(self.system.ccc_pool, "hebbian_update_lateral", None)
        if callable(updater):
            updater(fired_indices)

    def _replay_pool_concepts(self) -> int:
        replay = getattr(self.system.ccc_pool, "replay_exemplars", None)
        if not callable(replay):
            return 0
        return int(replay())

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
            return torch.zeros(self._concept_dim(), dtype=torch.float32), 0.0
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
        *,
        preview: bool = False,
    ) -> tuple[torch.Tensor, float, bool, float]:
        active_cccs: list[tuple[int, torch.Tensor, float]] = []
        for position, index in enumerate(fired_indices):
            confidence = (
                float(winner_confidences[position].item())
                if position < winner_confidences.numel()
                else 0.0
            )
            active_cccs.append((index, self._ccc_direction(index), confidence))

        vote_result, broadcast = self.system.workspace_consensus(
            active_cccs,
            update_workspace=not preview,
        )
        concept_direction = vote_result.winning_direction.detach().clone()
        return (
            concept_direction,
            float(vote_result.confidence),
            vote_result.voter_count == 0,
            float(self.system.normalized_broadcast_strength(broadcast)),
        )

    def _step_pool(
        self,
        tensor: torch.Tensor,
        *,
        allow_recruit: bool,
        learning_rate_multiplier: float = 1.0,
        preview: bool = False,
    ) -> PoolStepResult:
        pool = self.system.ccc_pool
        if preview:
            if not hasattr(pool, "preview"):
                raise AttributeError("CCC pool preview path is unavailable.")
            pool_output = pool.preview(tensor)
        else:
            pool_output = self.system._run_pool(
                tensor,
                allow_recruit=allow_recruit,
                learning_rate_multiplier=learning_rate_multiplier,
            )

        fired_indices = list(pool_output.fired_indices)
        winner_confidences = pool_output.winner_confidences.to(torch.float32)

        broadcast_strength = 0.0
        if getattr(self.system.config, "workspace", None) is not None:
            concept, confidence, abstained, broadcast_strength = self._workspace_consensus(
                fired_indices,
                winner_confidences,
                preview=preview,
            )
        else:
            concept, confidence = self._pool_concept(fired_indices, winner_confidences)
            abstained = len(fired_indices) == 0

        if not preview:
            self.system.timestep += 1
            self.last_fired_indices = fired_indices

        return PoolStepResult(
            fired_indices=fired_indices,
            concept_direction=concept,
            confidence=float(confidence),
            abstained=bool(abstained),
            winner_confidences=winner_confidences.detach().clone(),
            broadcast_strength=float(broadcast_strength),
        )

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

    def _best_same_label_ccc(
        self,
        concept_direction: torch.Tensor,
        label: int,
    ) -> tuple[int | None, float]:
        best_index: int | None = None
        best_similarity = float("-inf")
        for index, counts in self.ccc_label_counts.items():
            if not counts:
                continue
            dominant_label, _ = counts.most_common(1)[0]
            if int(dominant_label) != int(label):
                continue
            candidate = self._ccc_direction(index)
            similarity = float(cosine_similarity(candidate, concept_direction).item())
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index is None:
            return None, 0.0
        return best_index, best_similarity

    def _supervised_conv_merge(
        self,
        tensor: torch.Tensor,
        label: int | None,
        *,
        learning_rate_multiplier: float,
    ) -> tuple[int | None, bool, bool, float, PoolStepResult] | None:
        if not self.config.use_conv_ccc or label is None:
            return None
        if float(self.config.supervised_merge_threshold) <= 0.0:
            return None
        pool = self.system.ccc_pool
        if not isinstance(pool, ConvCCCPool):
            return None

        raw_batch, f1_batch, f1_trace, _ = pool._encode_shared_f1(tensor)  # noqa: SLF001
        concept_direction = normalize(f1_batch.mean(dim=0, keepdim=True)).squeeze(0)
        candidate_index, similarity = self._best_same_label_ccc(concept_direction, int(label))
        if candidate_index is None or similarity < float(self.config.supervised_merge_threshold):
            return None

        candidate = pool.cccs[candidate_index]
        if bool(candidate.is_locked.item()):
            return None

        candidate.learn_slow(
            raw_batch,
            f1_batch,
            ResonanceOutput(
                match_score=torch.full((raw_batch.shape[0],), similarity, dtype=torch.float32, device=raw_batch.device),
                resonated=torch.ones(raw_batch.shape[0], dtype=torch.bool, device=raw_batch.device),
                learn_signal=torch.ones(raw_batch.shape[0], dtype=torch.float32, device=raw_batch.device),
            ),
            f1_trace=f1_trace,
            timestep=int(self.system.timestep),
            learning_rate_multiplier=learning_rate_multiplier,
        )
        candidate.last_fired.fill_(int(self.system.timestep))
        self.system.timestep += 1
        self.last_fired_indices = [candidate_index]
        winner_confidences = torch.tensor([similarity], dtype=torch.float32)
        return (
            int(label),
            False,
            False,
            similarity,
            PoolStepResult(
                fired_indices=[candidate_index],
                concept_direction=self._ccc_direction(candidate_index),
                confidence=similarity,
                abstained=False,
                winner_confidences=winner_confidences,
                broadcast_strength=0.0,
            ),
        )

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

    def _preview_novelty_signal(self, prediction_error: float) -> NoveltySignal | None:
        if self.curiosity_system is None:
            return None

        prediction_error = abs(float(prediction_error))
        system = self.curiosity_system
        baseline = (
            max(float(system.prediction_error_baseline.item()), system.eps)
            if int(system._prediction_error_count.item()) > 0
            else max(prediction_error, system.eps)
        )
        novelty_ratio = prediction_error / baseline
        novelty_score = max(0.0, novelty_ratio - 1.0)
        is_novel = bool(
            prediction_error > 0.0
            and novelty_ratio > float(system.config.novelty_threshold)
        )
        novelty_state = 1.0 if is_novel else float(system.novelty_state.item())
        learning_boost = 1.0 + (
            max(0.0, float(system.config.novelty_boost) - 1.0) * novelty_state
        )
        return NoveltySignal(
            is_novel=is_novel,
            novelty_score=float(novelty_score),
            orienting_response=is_novel,
            learning_boost=float(max(1.0, learning_boost)),
            attention_disruption=float(max(0.0, min(1.0, novelty_state))),
        )

    @staticmethod
    def _boundary_score(winner_confidences: torch.Tensor) -> float:
        if winner_confidences.numel() < 2:
            return 0.0
        top_values, _ = torch.topk(winner_confidences.to(torch.float32), k=2)
        return float(max(0.0, min(1.0, 1.0 - abs(float(top_values[0] - top_values[1])))))

    def _curiosity_replay_count(
        self,
        *,
        reward_step,
        prediction: int | None,
        label: int | None,
        abstained: bool,
        recruited: bool,
        replay_depth: int,
        boundary_score: float = 0.0,
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
        effective_weight = float(self.config.curiosity_weight)
        if self.config.contrastive_curiosity:
            priority += max(0.0, boundary_score)
            effective_weight *= 1.0 + max(0.0, boundary_score)
        if priority * effective_weight < 0.75:
            return 0
        replay_count = 1
        if self.config.contrastive_curiosity and (boundary_score * effective_weight) >= 0.5:
            replay_count += 1
        return replay_count

    def _train_single_sample(
        self,
        tensor: torch.Tensor,
        label: int | None,
        *,
        learning_rate_multiplier: float = 1.0,
    ) -> tuple[int | None, bool, bool, float, PoolStepResult]:
        before_committed = int(self._pool_stats()["num_committed"])
        tensor = self._prepare_tensor(tensor)
        self._observe_pool_samples(1)
        merged_result = self._supervised_conv_merge(
            tensor,
            label,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        if merged_result is not None:
            prediction, abstained_flag, recruited, confidence, step_result = merged_result
        else:
            step_result = self._step_pool(
                tensor,
                allow_recruit=True,
                learning_rate_multiplier=learning_rate_multiplier,
            )
            prediction = (
                None
                if step_result.abstained
                else self._recognition_label(step_result.concept_direction, step_result.fired_indices)
            )
            abstained_flag = bool(step_result.abstained)
            recruited = int(self._pool_stats()["num_committed"]) > before_committed
            confidence = float(step_result.confidence)
        if not step_result.abstained and label is not None:
            self.label_bank.update(int(label), step_result.concept_direction)
        self._update_pool_importance(step_result.fired_indices, step_result.winner_confidences)
        self._update_lateral_predictions(step_result.fired_indices)
        self._record_ccc_activity(step_result.fired_indices, label, step_result.confidence)
        hebbian_update = getattr(self.system.ccc_pool, "hebbian_update", None)
        if callable(hebbian_update) and merged_result is None:
            hebbian_update(
                tensor,
                fired_indices=step_result.fired_indices,
                winner_confidences=step_result.winner_confidences,
                recruited=recruited,
                learning_rate_multiplier=learning_rate_multiplier,
            )
        return (
            prediction,
            abstained_flag,
            recruited,
            confidence,
            step_result,
        )

    @torch.no_grad()
    def train_online(
        self,
        data_stream: StreamingDataSource | Iterable[DataSample | tuple[torch.Tensor, int]] | None = None,
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
        self._flush_pool_hebbian_updates()
        auto_conv_benchmark = data_stream is None and bool(self.config.use_conv_ccc)
        num_passes = max(1, int(num_passes))
        if data_stream is None:
            data_stream, _, _ = load_cifar10_or_synthetic(
                train_samples=target_samples,
                test_samples=self.config.num_test_samples,
                seed=0,
                augmentation=self.config.augmentation,
            )
            if self.config.augmentation.enabled:
                target_samples = len(data_stream)
        if auto_conv_benchmark and num_passes == 1:
            num_passes = 3
        samples = self._materialize_samples(data_stream, target_samples)
        train_samples, warmup_samples = self._fit_preprocessing(samples)
        if warmup_samples > 0:
            self._observe_pool_samples(warmup_samples)
            if warmup_samples >= int(getattr(self.system.config.ccc, "freeze_f1_after", 0)):
                self._freeze_pool_f1_if_ready()

        if interleave_classes:
            train_samples = _interleave_by_class(train_samples)

        if self.curiosity_system is not None:
            self.curiosity_system.reset()
        if self.curriculum_scheduler is not None:
            self.curriculum_scheduler.difficulty_scores.clear()

        indexed_train_samples = [(sample_id, tensor, label) for sample_id, (tensor, label) in enumerate(train_samples)]
        effective_target = len(indexed_train_samples) * num_passes
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
        boundary_scores: list[float] = []
        learning_rate_multipliers: list[float] = []
        curiosity_replays = 0
        concept_replay_events = 0
        concept_replay_boosts = 0
        maturation_transitions: list[dict[str, float | int]] = []
        progress_interval = max(1, max(effective_target, 1) // 10)
        replay_interval = int(getattr(self.system.config.ccc, "replay_interval", 0))
        replay_enabled = bool(getattr(self.system.config.ccc, "enable_replay", False))
        primary_processed = 0

        for pass_index in range(num_passes):
            pass_lr_scale = float(max(0.0, self.config.pass_lr_decay)) ** pass_index
            if pass_index == 0:
                pass_samples = indexed_train_samples
            elif self.curriculum_scheduler is not None:
                ordered_ids = self.curriculum_scheduler.order_samples(
                    [sample_id for sample_id, _, _ in indexed_train_samples]
                )
                sample_lookup = {
                    sample_id: (tensor, label)
                    for sample_id, tensor, label in indexed_train_samples
                }
                pass_samples = [
                    (sample_id, *sample_lookup[sample_id])
                    for sample_id in ordered_ids
                ]
            else:
                # Shuffle independently for each repeat pass so the system
                # does not memorise a fixed sequence.
                perm = torch.randperm(len(indexed_train_samples)).tolist()
                pass_samples = [indexed_train_samples[i] for i in perm]

            queue = deque((sample_id, tensor, label, 0) for sample_id, tensor, label in pass_samples)
            while queue:
                sample_id, tensor, label, replay_depth = queue.popleft()
                preview_result: PoolStepResult | None = None
                preview_prediction: int | None = None
                preview_abstained = False
                preview_confidence = 0.0
                prediction_error_preview = 0.0
                boundary_score = 0.0
                learning_rate_multiplier = pass_lr_scale
                precision_enabled = getattr(self.system.ccc_pool, "precision_gate", None) is not None

                needs_preview = (
                    precision_enabled
                    or self.curiosity_system is not None
                    or self.curriculum_scheduler is not None
                    or self.config.contrastive_curiosity
                    or getattr(self.system.config, "workspace", None) is not None
                )
                if needs_preview:
                    preview_result = self._step_pool(
                        self._prepare_tensor(tensor),
                        allow_recruit=False,
                        preview=True,
                    )
                    preview_abstained = preview_result.abstained
                    preview_confidence = float(preview_result.confidence)
                    preview_prediction = (
                        None
                        if preview_result.abstained
                        else self._recognition_label(
                            preview_result.concept_direction,
                            preview_result.fired_indices,
                        )
                    )
                    if self.curriculum_scheduler is not None and replay_depth == 0 and pass_index == 0:
                        self.curriculum_scheduler.score_sample(sample_id, preview_confidence)
                    if precision_enabled:
                        learning_rate_multiplier *= self._current_pool_precision()
                    if self.curiosity_system is not None:
                        prediction_error_preview = self._estimate_prediction_error(
                            prediction=preview_prediction,
                            label=label,
                            confidence=preview_confidence,
                            abstained=preview_abstained,
                            recruited=False,
                        )
                        novelty_signal = self._preview_novelty_signal(prediction_error_preview)
                        if novelty_signal is not None:
                            learning_rate_multiplier *= max(1.0, float(novelty_signal.learning_boost))
                    if getattr(self.system.config, "workspace", None) is not None:
                        gnw_learning_gain = max(
                            0.0,
                            float(getattr(self.system.gnw.config, "gnw_learning_gain", 0.75)),
                        )
                        learning_rate_multiplier *= (
                            1.0
                            + (gnw_learning_gain * (1.0 - float(preview_result.broadcast_strength)))
                        )
                    if self.config.contrastive_curiosity:
                        boundary_score = self._boundary_score(preview_result.winner_confidences)
                        boundary_scores.append(boundary_score)
                if self.maturation is not None:
                    learning_rate_multiplier *= float(self.maturation.get_learning_rate_scale())
                learning_rate_multipliers.append(float(learning_rate_multiplier))

                prediction, abstained_flag, recruited, confidence, step_result = self._train_single_sample(
                    tensor,
                    label,
                    learning_rate_multiplier=learning_rate_multiplier,
                )

                processed += 1
                if label is not None:
                    labeled += 1
                    correct += int(prediction == label)
                abstained += int(abstained_flag)
                if replay_depth == 0:
                    primary_processed += 1
                if self.maturation is not None:
                    previous_phase = self.maturation.phase
                    if self.maturation.check_transition(
                        step_result.winner_confidences.to(torch.float32).reshape(-1).tolist()
                    ):
                        self._apply_maturation_phase()
                        variance = (
                            0.0
                            if self.maturation.last_transition_variance is None
                            else float(self.maturation.last_transition_variance)
                        )
                        transition = {
                            "from_phase": previous_phase,
                            "to_phase": self.maturation.phase,
                            "variance": variance,
                            "processed_samples": processed,
                        }
                        maturation_transitions.append(transition)
                        print(
                            "[Maturation] "
                            f"Phase {previous_phase}\N{RIGHTWARDS ARROW}"
                            f"{self.maturation.phase}: "
                            f"representations stabilized (variance={variance:.2f})"
                        )

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
                        boundary_score=boundary_score,
                    )
                    curiosity_replays += replay_count
                    for _ in range(replay_count):
                        queue.append((sample_id, tensor.detach().clone(), label, replay_depth + 1))

                if (
                    replay_enabled
                    and replay_depth == 0
                    and replay_interval > 0
                    and primary_processed % replay_interval == 0
                ):
                    concept_replay_events += 1
                    concept_replay_boosts += self._replay_pool_concepts()

                pool_stats = self._pool_stats()
                current_capacity = max(int(pool_stats["total_concepts"]), 1)
                accuracy_curve.append(correct / max(labeled, 1))
                if replay_depth == 0:
                    raw_accuracy_curve.append(correct / max(labeled, 1))
                utilization_curve.append(int(pool_stats["num_committed"]) / current_capacity)
                abstention_curve.append(abstained / processed)

                if processed % progress_interval == 0 or processed == effective_target:
                    pass_tag = f"p{pass_index + 1}/{num_passes} " if num_passes > 1 else ""
                    print(
                        f"[train] {pass_tag}{processed}/{effective_target} "
                        f"cccs={int(pool_stats['num_committed'])} "
                        f"acc={accuracy_curve[-1]:.3f} "
                        f"abstain={abstention_curve[-1]:.3f}"
                    )
            self._flush_pool_hebbian_updates()

        self._flush_pool_hebbian_updates()
        result = {
            "processed_samples": processed,
            "raw_samples_seen": len(samples),
            "warmup_samples": warmup_samples,
            "num_passes": num_passes,
            "accuracy": correct / max(labeled, 1),
            "online_accuracy": correct / max(labeled, 1),
            "abstention_rate": abstained / max(processed, 1),
            "committed_cccs": int(self._pool_stats()["num_committed"]),
            "accuracy_curve": accuracy_curve,
            "raw_accuracy_curve": raw_accuracy_curve,
            "pool_utilization_curve": utilization_curve,
            "abstention_rate_curve": abstention_curve,
            "curiosity_enabled": self.curiosity_system is not None,
            "curriculum_enabled": self.curriculum_scheduler is not None,
            "contrastive_curiosity_enabled": bool(self.config.contrastive_curiosity),
            "curiosity_replays": curiosity_replays,
            "concept_replay_events": concept_replay_events,
            "concept_replay_boosts": concept_replay_boosts,
            "mean_prediction_error": sum(prediction_errors) / max(len(prediction_errors), 1),
            "mean_novelty": sum(novelty_scores) / max(len(novelty_scores), 1),
            "mean_boundary_score": sum(boundary_scores) / max(len(boundary_scores), 1),
            "mean_learning_rate_multiplier": (
                sum(learning_rate_multipliers) / max(len(learning_rate_multipliers), 1)
            ),
            "peak_curiosity_drive": max(curiosity_drives, default=0.0),
            "maturation_enabled": self.maturation is not None,
            "maturation_phase": 1 if self.maturation is None else int(self.maturation.phase),
            "maturation_transitions": maturation_transitions,
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
        self._flush_pool_hebbian_updates()
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
            step_result = self._step_pool(tensor, allow_recruit=False)
            prediction = (
                None
                if step_result.abstained
                else self._recognition_label(step_result.concept_direction, step_result.fired_indices)
            )

            total += 1
            labeled += int(label is not None)
            correct += int(label is not None and prediction == label)
            covered += int(prediction is not None)
            covered_correct += int(label is not None and prediction == label and prediction is not None)
            abstained += int(step_result.abstained)
            firing_sum += len(step_result.fired_indices)
            predictions.append(prediction)
            labels.append(label)
            if label is not None:
                per_class_totals[int(label)] += 1
                per_class_correct[int(label)] += int(prediction == label)

        committed = max(int(self._pool_stats()["num_committed"]), 1)
        current_capacity = max(self._current_pool_capacity(), 1)
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
            "pool_utilization": committed / current_capacity,
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
        self._flush_pool_hebbian_updates()
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
            step_result = self._step_pool(
                self._prepare_tensor(sample.to(torch.float32).reshape(-1)),
                allow_recruit=False,
            )
            total += 1
            abstained += int(step_result.abstained or step_result.confidence < ood_threshold)
            mean_firing += len(step_result.fired_indices)
            mean_confidence += step_result.confidence

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
                max_growth_factor=self.config.max_growth_factor,
                margin_threshold=max(self.config.margin_threshold, 0.55),
                use_batched=self.config.use_batched,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                consolidation_strength=self.config.consolidation_strength,
                freeze_f1_after=self.config.freeze_f1_after,
                f1_adapter_dim=self.config.f1_adapter_dim,
                num_train_samples=self.config.num_train_samples,
                num_test_samples=self.config.num_test_samples,
                preprocessing_warmup_samples=self.config.preprocessing_warmup_samples,
                curiosity_weight=self.config.curiosity_weight,
                curriculum=self.config.curriculum,
                contrastive_curiosity=self.config.contrastive_curiosity,
                workspace=copy.deepcopy(self.config.workspace),
                maturation=copy.deepcopy(self.config.maturation),
                augmentation=copy.deepcopy(self.config.augmentation),
            ),
            preprocessing=copy.deepcopy(self.preprocessing),
        )
        class_sets = [set(int(label) for label in group) for group in class_order]
        stage_train_limits = max(200, self.config.num_train_samples // max(len(class_sets), 1))
        eval_limit = max(100, self.config.num_test_samples // max(len(class_sets[0]), 1))

        old_classes = class_sets[0]
        old_train = take_samples(stream, stage_train_limits, allowed_labels=old_classes)
        old_eval = take_samples(stream, eval_limit, allowed_labels=old_classes)
        shadow.start_new_task()
        shadow.train_online(old_train, num_samples=len(old_train))
        before = shadow.evaluate(old_eval, num_samples=len(old_eval))

        stage_metrics: list[dict[str, object]] = [before]
        for stage_labels in class_sets[1:]:
            stage_train = take_samples(stream, stage_train_limits, allowed_labels=stage_labels)
            shadow.start_new_task()
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
