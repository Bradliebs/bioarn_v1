"""Scaling-law sweeps for Bio-ARN on CIFAR-10."""

from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
from dataclasses import dataclass
import io
from itertools import chain
import json
from pathlib import Path
import math
import sys
import time

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, PredictiveConfig, SDMConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool, ExpertConfig
from bioarn.hardware.energy_model import EnergyModel
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.preprocessing import CompetitiveLearner, OnlinePCA, PreprocessingPipeline, SparseRandomProjection
from bioarn.scaling import BatchedCCCPool
from bioarn.training import (
    EnsembleTrainer,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

POOL_SIZES = [50, 100, 200, 400, 800]
HIERARCHY_DEPTHS = [2, 3, 4]
EXPERT_COUNTS = [3, 5, 7, 10]
DATA_VOLUMES = [100, 200, 500, 1000, 2000]

DEFAULT_TRAIN_SAMPLES = 400
TEST_SAMPLES = 200
OOD_SAMPLES = 200
SEED = 11
OOD_SEED = 19


@dataclass
class SweepResult:
    sweep: str
    label: str
    parameter: float
    accuracy: float
    train_seconds: float
    energy_joules: float
    ood_auroc: float
    active_cccs: float
    source: str
    model: str

    @property
    def efficiency(self) -> float:
        return self.accuracy / max(self.train_seconds * self.energy_joules, 1e-12)


def _auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    positives = [(score, 1) for score in id_scores]
    negatives = [(score, 0) for score in ood_scores]
    ranked = sorted(positives + negatives, key=lambda item: item[0], reverse=True)
    total_pos = len(id_scores)
    total_neg = len(ood_scores)
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    prev_score: float | None = None
    area = 0.0
    for score, label in ranked:
        if prev_score is not None and score != prev_score:
            area += (fp - prev_fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    area += (fp - prev_fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
    area += (total_neg - fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
    return float(area)


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    buckets: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
        else:
            buckets[int(label)].append((tensor, label))

    ordered: list[tuple[torch.Tensor, int | None]] = []
    labels = sorted(buckets)
    while any(buckets[label] for label in labels):
        for label in labels:
            if buckets[label]:
                ordered.append(buckets[label].pop(0))
    ordered.extend(unlabeled)
    return ordered


def _make_ood_samples(num_samples: int, *, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(3072, generator=generator) for _ in range(num_samples)]


def _format_table(results: list[SweepResult]) -> str:
    headers = ["config", "acc", "train_s", "energy_j", "ood_auroc", "active_cccs"]
    rows = [
        [
            result.label,
            f"{result.accuracy:.3f}",
            f"{result.train_seconds:.2f}",
            f"{result.energy_joules:.6f}",
            f"{result.ood_auroc:.3f}",
            f"{result.active_cccs:.2f}",
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    body = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([header_line, divider, *body])


def _fit_r2(xs: list[float], ys: list[float], transform) -> float:
    transformed = torch.tensor([transform(value) for value in xs], dtype=torch.float64)
    targets = torch.tensor(ys, dtype=torch.float64)
    if transformed.numel() < 2:
        return 1.0
    x_centered = transformed - transformed.mean()
    denom = float((x_centered * x_centered).sum().item())
    if denom <= 1e-12:
        return 0.0
    slope = float((x_centered * (targets - targets.mean())).sum().item()) / denom
    intercept = float(targets.mean().item()) - (slope * float(transformed.mean().item()))
    predictions = (slope * transformed) + intercept
    sst = float(((targets - targets.mean()) ** 2).sum().item())
    if sst <= 1e-12:
        return 1.0
    sse = float(((targets - predictions) ** 2).sum().item())
    return 1.0 - (sse / sst)


def _trend_label(results: list[SweepResult]) -> str:
    xs = [result.parameter for result in results]
    ys = [result.accuracy for result in results]
    fits = {
        "linear": _fit_r2(xs, ys, lambda value: value),
        "logarithmic": _fit_r2(xs, ys, lambda value: math.log(max(value, 1.0))),
        "sublinear": _fit_r2(xs, ys, lambda value: math.sqrt(max(value, 0.0))),
    }
    return max(fits.items(), key=lambda item: item[1])[0]


def _diminishing_returns(results: list[SweepResult]) -> str:
    if len(results) < 2:
        return results[0].label
    gains: list[float] = []
    slopes: list[float] = []
    for previous, current in zip(results, results[1:], strict=False):
        gain = current.accuracy - previous.accuracy
        gains.append(gain)
        delta = max(current.parameter - previous.parameter, 1e-9)
        slopes.append(gain / delta)

    baseline = max(slopes[0], 1e-9)
    for index, (result, gain, slope) in enumerate(zip(results[1:], gains, slopes, strict=False), start=1):
        if gain <= 0.005 or slope <= baseline * 0.35:
            return result.label
        if index >= 2 and gain <= max(gains[index - 1], 0.0) * 0.6:
            return result.label
    return results[-1].label


def _estimate_total_energy(
    config: BioARNConfig,
    *,
    active_cccs: float,
    train_samples: int,
    eval_samples: int,
    ood_samples: int,
) -> float:
    energy_model = EnergyModel()
    active = max(1, int(round(active_cccs)))
    train_energy = energy_model.estimate_learning_energy(config, "loihi2").total_joules * train_samples
    infer_energy = energy_model.estimate_inference_energy(config, "loihi2", active).total_joules
    return float(train_energy + (infer_energy * (eval_samples + ood_samples)))


def _vision_base_config(*, max_pool_size: int, num_train_samples: int) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=64,
        max_pool_size=max_pool_size,
        margin_threshold=0.38,
        use_batched=True,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=num_train_samples,
        num_test_samples=TEST_SAMPLES,
        preprocessing_warmup_samples=64,
    )


def _vision_scores(
    trainer: VisionTrainer,
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> tuple[list[float], list[float]]:
    id_scores: list[float] = []
    for tensor, _ in test_samples:
        _, _, confidence, _ = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        _, _, confidence, _ = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(confidence))
    return id_scores, ood_scores


def _ensemble_signal_strength(confidence: float, agreement: float, abstention_fraction: float) -> float:
    agreement_bonus = 0.7 + (0.3 * agreement)
    abstention_penalty = 1.0 - (0.35 * abstention_fraction)
    return float(max(0.0, min(1.0, confidence * agreement_bonus * abstention_penalty)))


def _ensemble_proxy_config(ensemble: EnsemblePool) -> BioARNConfig:
    first_pool = ensemble.experts[0].pool
    pool_config = first_pool.config
    total_pool_size = 0
    for expert in ensemble.experts:
        pool = expert.pool
        if hasattr(pool.config, "max_pool_size"):
            total_pool_size += int(pool.config.max_pool_size)
        elif hasattr(pool.config, "pool_sizes"):
            total_pool_size += int(sum(pool.config.pool_sizes))
    concept_dim = int(getattr(pool_config, "concept_dim", 128))
    return BioARNConfig(
        ccc=CCCConfig(
            input_dim=int(pool_config.input_dim),
            concept_dim=concept_dim,
            num_f1_features=int(pool_config.num_f1_features),
            f1_top_k=int(pool_config.f1_top_k),
            fast_lr=float(pool_config.fast_lr),
            slow_lr=float(pool_config.slow_lr),
            feedback_lr=float(pool_config.feedback_lr),
            max_pool_size=total_pool_size,
        ),
        sdm=SDMConfig(
            address_dim=max(512, concept_dim * 4),
            hamming_radius=max(16, concept_dim // 4),
            num_hard_locations=256,
            data_dim=concept_dim,
            decay_rate=0.999,
            stdp_window=10,
        ),
        predictive=PredictiveConfig(num_levels=1, settling_steps=4),
        gnw=GNWConfig(capacity=min(7, len(ensemble.experts) + 1), concept_dim=concept_dim),
        seed=SEED,
    )


def _hierarchy_proxy_config(hierarchy: VisualHierarchy, depth: int) -> BioARNConfig:
    active_layers = hierarchy.layers[:depth]
    mean_input = int(round(sum(layer.input_dim for layer in active_layers) / len(active_layers)))
    mean_concept = int(round(sum(layer.concept_dim for layer in active_layers) / len(active_layers)))
    mean_features = int(
        round(sum(layer.pool.config.num_f1_features for layer in active_layers) / len(active_layers))
    )
    total_pool_size = sum(int(layer.pool.config.max_pool_size) for layer in active_layers)
    return BioARNConfig(
        ccc=CCCConfig(
            input_dim=mean_input,
            concept_dim=mean_concept,
            num_f1_features=mean_features,
            f1_top_k=max(4, min(32, mean_features // 4)),
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=total_pool_size,
        ),
        sdm=SDMConfig(
            address_dim=max(512, mean_concept * 4),
            hamming_radius=max(16, mean_concept // 4),
            num_hard_locations=256,
            data_dim=mean_concept,
            decay_rate=0.999,
            stdp_window=10,
        ),
        predictive=PredictiveConfig(num_levels=depth, settling_steps=4),
        gnw=GNWConfig(capacity=min(7, depth + 2), concept_dim=mean_concept),
        seed=SEED,
    )


class DepthLimitedHierarchyRunner:
    """Trains and evaluates only the first N levels of a VisualHierarchy."""

    def __init__(self, depth: int, *, pool_size: int = 200) -> None:
        if depth not in {2, 3, 4}:
            raise ValueError("depth must be one of {2, 3, 4}.")
        self.depth = depth
        self.hierarchy = VisualHierarchy(
            HierarchyConfig(
                image_size=(32, 32, 3),
                patch_sizes=[8, 2, 1, 1],
                pool_sizes=[pool_size, pool_size, pool_size, pool_size],
                concept_dims=[16, 32, 48, 32],
                thresholds=[0.24, 0.28, 0.32, 0.36],
                learning_rates=[0.05, 0.03, 0.02, 0.01],
                class_count=10,
                predictive=None,
            )
        )
        self.label_prototypes: dict[int, torch.Tensor] = {}
        self.label_counts: Counter[int] = Counter()
        self.ccc_label_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)

    def _prepare_patches(self, image: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, int]]:
        patches = self.hierarchy.extractor.extract_patches(
            image,
            patch_size=self.hierarchy.config.patch_size,
            stride=self.hierarchy.config.patch_size,
        )
        if self.hierarchy.attention is not None and patches:
            frame = self.hierarchy.extractor._ensure_image(image)  # noqa: SLF001
            gains = self.hierarchy.attention.patch_gains(
                frame,
                self.hierarchy.extractor.last_patch_positions,
                patch_size=self.hierarchy.config.patch_size,
            )
            patches = self.hierarchy.attention.apply_to_patches(
                patches,
                gains,
                sensory_dim=(
                    self.hierarchy.config.patch_size
                    * self.hierarchy.config.patch_size
                    * self.hierarchy.config.channels
                ),
            )
        return patches, self.hierarchy.extractor.last_grid_shape

    def _run_to_depth(
        self,
        image: torch.Tensor,
        *,
        learn: bool,
        label: int | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[list[int]]], list[torch.Tensor]]:
        patches, patch_grid = self._prepare_patches(image)
        layer_inputs: list[torch.Tensor] = []
        layer_activations: list[torch.Tensor] = []
        layer_fired: list[list[list[int]]] = []
        layer_confidences: list[torch.Tensor] = []

        l1_inputs = self.hierarchy._stack(patches, self.hierarchy.config.l1_input_dim)  # noqa: SLF001
        layer_inputs.append(l1_inputs)
        l1_result = (
            self.hierarchy._run_layer_learn(self.hierarchy.layers[0], l1_inputs, allow_recruit=True)  # noqa: SLF001
            if learn
            else self.hierarchy._run_layer_infer(self.hierarchy.layers[0], l1_inputs)  # noqa: SLF001
        )
        layer_activations.append(l1_result.activations)
        layer_fired.append(l1_result.fired_indices)
        layer_confidences.append(l1_result.confidences)

        l2_inputs, _ = self.hierarchy._prepare_l2_inputs(l1_result.activations, patch_grid)  # noqa: SLF001
        layer_inputs.append(l2_inputs)
        l2_result = (
            self.hierarchy._run_layer_learn(  # noqa: SLF001
                self.hierarchy.layers[1],
                l2_inputs,
                allow_recruit=self.depth > 2 or label is not None,
            )
            if learn
            else self.hierarchy._run_layer_infer(self.hierarchy.layers[1], l2_inputs)  # noqa: SLF001
        )
        layer_activations.append(l2_result.activations)
        layer_fired.append(l2_result.fired_indices)
        layer_confidences.append(l2_result.confidences)
        if self.depth == 2:
            return layer_inputs, layer_activations, layer_fired, layer_confidences

        l3_inputs, _ = self.hierarchy._prepare_l3_inputs(l2_result.activations)  # noqa: SLF001
        layer_inputs.append(l3_inputs)
        l3_result = (
            self.hierarchy._run_layer_learn(  # noqa: SLF001
                self.hierarchy.layers[2],
                l3_inputs,
                allow_recruit=self.depth > 3 or label is not None,
            )
            if learn
            else self.hierarchy._run_layer_infer(self.hierarchy.layers[2], l3_inputs)  # noqa: SLF001
        )
        layer_activations.append(l3_result.activations)
        layer_fired.append(l3_result.fired_indices)
        layer_confidences.append(l3_result.confidences)
        if self.depth == 3:
            return layer_inputs, layer_activations, layer_fired, layer_confidences

        l4_inputs, _ = self.hierarchy._prepare_l4_inputs(l3_result.activations)  # noqa: SLF001
        layer_inputs.append(l4_inputs)
        l4_result = (
            self.hierarchy._run_layer_learn(  # noqa: SLF001
                self.hierarchy.layers[3],
                l4_inputs,
                allow_recruit=label is not None,
            )
            if learn
            else self.hierarchy._run_layer_infer(self.hierarchy.layers[3], l4_inputs)  # noqa: SLF001
        )
        layer_activations.append(l4_result.activations)
        layer_fired.append(l4_result.fired_indices)
        layer_confidences.append(l4_result.confidences)
        return layer_inputs, layer_activations, layer_fired, layer_confidences

    @staticmethod
    def _flatten_indices(fired_indices: list[list[int]]) -> list[int]:
        return sorted(set(chain.from_iterable(fired_indices)))

    def _feature_signature(
        self,
        layer_inputs: list[torch.Tensor],
        layer_activations: list[torch.Tensor],
    ) -> torch.Tensor:
        parts = [layer_inputs[0].reshape(-1)]
        parts.extend(activations.reshape(-1) for activations in layer_activations)
        return self.hierarchy._normalize_feature(torch.cat(parts, dim=0))  # noqa: SLF001

    def _has_label_specialist(self, label: int, fired_indices: list[int]) -> bool:
        for fired_index in fired_indices:
            counts = self.ccc_label_counts.get(int(fired_index))
            if not counts:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / max(sum(counts.values()), 1)
            if int(dominant_label) == int(label) and purity >= 0.55:
                return True
        return False

    def _update_label_memory(
        self,
        label: int,
        evidence_feature: torch.Tensor,
        fired_indices: list[int],
    ) -> None:
        if float(evidence_feature.norm().item()) <= 1e-8:
            return
        normalized = self.hierarchy._normalize_feature(evidence_feature)  # noqa: SLF001
        count = self.label_counts[int(label)] + 1
        if int(label) not in self.label_prototypes:
            self.label_prototypes[int(label)] = normalized.detach().clone()
        else:
            updated = (
                (self.label_prototypes[int(label)] * self.label_counts[int(label)]) + normalized
            ) / count
            self.label_prototypes[int(label)] = self.hierarchy._normalize_feature(updated)  # noqa: SLF001
        self.label_counts[int(label)] = count
        for fired_index in fired_indices:
            self.ccc_label_counts[int(fired_index)][int(label)] += 1

    def _prototype_prediction(self, evidence_feature: torch.Tensor) -> tuple[int | None, float]:
        if not self.label_prototypes or float(evidence_feature.norm().item()) <= 1e-8:
            return None, 0.0
        labels = sorted(self.label_prototypes)
        stacked = torch.stack(
            [self.label_prototypes[label].to(evidence_feature) for label in labels],
            dim=0,
        )
        query = self.hierarchy._normalize_feature(evidence_feature).unsqueeze(0).expand_as(stacked)  # noqa: SLF001
        similarities = cosine_similarity(stacked, query)
        best_index = int(torch.argmax(similarities).item())
        return int(labels[best_index]), float(similarities[best_index].item())

    def _predict_label(
        self,
        evidence_feature: torch.Tensor,
        fired_indices: list[int],
        confidence: float,
    ) -> tuple[int, float]:
        votes: defaultdict[int, float] = defaultdict(float)
        for fired_index in fired_indices:
            counts = self.ccc_label_counts.get(int(fired_index))
            if not counts:
                continue
            total = sum(counts.values())
            for label, count in counts.items():
                votes[int(label)] += count / max(total, 1)

        if votes:
            voted_label, vote_strength = max(votes.items(), key=lambda item: item[1])
            return int(voted_label), max(float(confidence), float(vote_strength / max(len(fired_indices), 1)))

        prototype_label, prototype_similarity = self._prototype_prediction(evidence_feature)
        if prototype_label is None or prototype_similarity < 0.2:
            return -1, max(float(confidence), float(prototype_similarity))
        return int(prototype_label), max(float(confidence), float(prototype_similarity))

    def train(
        self,
        samples: list[tuple[torch.Tensor, int | None]],
    ) -> tuple[float, float]:
        total = correct = 0
        total_fired = 0.0
        for tensor, label in samples:
            self.hierarchy._update_structure_stats(tensor)  # noqa: SLF001
            layer_inputs, layer_activations, layer_fired, layer_confidences = self._run_to_depth(
                tensor,
                learn=True,
                label=label,
            )
            final_inputs = layer_inputs[-1]
            final_fired = self._flatten_indices(layer_fired[-1])
            if (
                label is not None
                and final_inputs.shape[0] > 0
                and not self._has_label_specialist(int(label), final_fired)
            ):
                recruited = self.hierarchy.layers[self.depth - 1].pool.recruit_single(
                    final_inputs[0],
                    timestep=self.hierarchy.timestep,
                )
                self.hierarchy.timestep += 1
                if recruited is not None:
                    final_fired = sorted(set(final_fired + recruited.fired_indices))

            evidence = self._feature_signature(layer_inputs, layer_activations)
            confidence = (
                float(layer_confidences[-1].max().item()) if layer_confidences[-1].numel() else 0.0
            )
            prediction, _ = self._predict_label(evidence, final_fired, confidence)
            if label is not None:
                self._update_label_memory(int(label), evidence, final_fired)
                correct += int(prediction == label)
                total += 1
            total_fired += sum(len(indices) for indices in layer_fired[-1])
        accuracy = correct / max(total, 1)
        mean_fired = total_fired / max(len(samples), 1)
        return accuracy, mean_fired

    def classify(self, tensor: torch.Tensor) -> tuple[int, float, int]:
        if self.hierarchy._is_noise_like(tensor):  # noqa: SLF001
            return -1, 0.0, 0
        layer_inputs, layer_activations, layer_fired, layer_confidences = self._run_to_depth(
            tensor,
            learn=False,
        )
        final_fired = self._flatten_indices(layer_fired[-1])
        confidence = float(layer_confidences[-1].max().item()) if layer_confidences[-1].numel() else 0.0
        evidence = self._feature_signature(layer_inputs, layer_activations)
        label, score = self._predict_label(evidence, final_fired, confidence)
        total_fired = sum(len(indices) for indices in layer_fired[-1])
        if label < 0:
            return -1, float(score), total_fired
        return int(label), float(min(1.0, score)), total_fired


def _evaluate_depth_runner(
    runner: DepthLimitedHierarchyRunner,
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> tuple[float, float, list[float], list[float]]:
    correct = 0
    total = 0
    total_fired = 0.0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        prediction, confidence, active = runner.classify(tensor)
        id_scores.append(float(confidence))
        total += 1
        total_fired += active
        if label is not None and prediction == label:
            correct += 1

    ood_scores: list[float] = []
    for tensor in ood_samples:
        _, confidence, _ = runner.classify(tensor)
        ood_scores.append(float(confidence))

    return correct / max(total, 1), total_fired / max(total, 1), id_scores, ood_scores


def _evaluate_ensemble(
    ensemble: EnsemblePool,
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> tuple[float, float, list[float], list[float]]:
    correct = 0
    total = 0
    total_fired = 0.0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        result = ensemble.classify(tensor)
        confidence = _ensemble_signal_strength(
            float(result.confidence),
            float(result.agreement),
            float(result.abstention_fraction),
        )
        id_scores.append(confidence)
        total += 1
        if label is not None and result.predicted_class == label:
            correct += 1

        fired = 0
        for expert in ensemble.experts:
            pool = expert.pool
            if isinstance(pool, BatchedCCCPool):
                transformed = ensemble._transform(expert, tensor)  # noqa: SLF001
                fired += len(ensemble._infer_batched_pool(pool, transformed).fired_indices)  # noqa: SLF001
            elif isinstance(pool, VisualHierarchy):
                output = pool.process(tensor)
                fired += sum(len(indices) for layer in output.fired_indices for indices in layer)
        total_fired += fired

    ood_scores: list[float] = []
    for tensor in ood_samples:
        result = ensemble.classify(tensor)
        ood_scores.append(
            _ensemble_signal_strength(
                float(result.confidence),
                float(result.agreement),
                float(result.abstention_fraction),
            )
        )
    return correct / max(total, 1), total_fired / max(total, 1), id_scores, ood_scores


def _load_data() -> tuple[list[tuple[torch.Tensor, int | None]], list[tuple[torch.Tensor, int | None]], str]:
    max_train = max(DATA_VOLUMES)
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=max_train,
        test_samples=TEST_SAMPLES,
        seed=SEED,
        timeout_seconds=5.0,
    )
    train_samples = take_samples(train_stream, max_train)
    test_samples = take_samples(test_stream, TEST_SAMPLES)
    return _interleave_by_class(train_samples), test_samples, source


def run_pool_size_sweep(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
    source: str,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    subset = train_samples[:DEFAULT_TRAIN_SAMPLES]
    for pool_size in POOL_SIZES:
        print(f"[pool_size] {pool_size}")
        trainer = VisionTrainer(_vision_base_config(max_pool_size=pool_size, num_train_samples=len(subset)))
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train_online(subset, num_samples=len(subset), interleave_classes=True)
        train_seconds = time.perf_counter() - start
        metrics = trainer.evaluate(test_samples, num_samples=TEST_SAMPLES)
        active_cccs = float(metrics["mean_firing_count"])
        id_scores, ood_scores = _vision_scores(trainer, test_samples, ood_samples)
        results.append(
            SweepResult(
                sweep="pool_size",
                label=f"pool={pool_size}",
                parameter=float(pool_size),
                accuracy=float(metrics["accuracy"]),
                train_seconds=float(train_seconds),
                energy_joules=_estimate_total_energy(
                    trainer.system.config,
                    active_cccs=active_cccs,
                    train_samples=len(subset),
                    eval_samples=TEST_SAMPLES,
                    ood_samples=len(ood_samples),
                ),
                ood_auroc=_auroc(id_scores, ood_scores),
                active_cccs=active_cccs,
                source=source,
                model="vision_trainer",
            )
        )
    return results


def run_depth_sweep(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
    source: str,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    subset = train_samples[:DEFAULT_TRAIN_SAMPLES]
    for depth in HIERARCHY_DEPTHS:
        print(f"[hierarchy_depth] {depth}")
        runner = DepthLimitedHierarchyRunner(depth, pool_size=100)
        start = time.perf_counter()
        runner.train(subset)
        train_seconds = time.perf_counter() - start
        accuracy, active_cccs, id_scores, ood_scores = _evaluate_depth_runner(runner, test_samples, ood_samples)
        proxy_config = _hierarchy_proxy_config(runner.hierarchy, depth)
        results.append(
            SweepResult(
                sweep="hierarchy_depth",
                label=f"depth={depth}",
                parameter=float(depth),
                accuracy=float(accuracy),
                train_seconds=float(train_seconds),
                energy_joules=_estimate_total_energy(
                    proxy_config,
                    active_cccs=active_cccs,
                    train_samples=len(subset),
                    eval_samples=TEST_SAMPLES,
                    ood_samples=len(ood_samples),
                ),
                ood_auroc=_auroc(id_scores, ood_scores),
                active_cccs=float(active_cccs),
                source=source,
                model="visual_hierarchy",
            )
        )
    return results


def run_expert_sweep(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
    source: str,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    subset = train_samples[:DEFAULT_TRAIN_SAMPLES]
    manager = DiversityManager()

    def make_pool(*, input_dim: int, threshold: float, pool_size: int = 96) -> BatchedCCCPool:
        return manager._build_pool(  # noqa: SLF001
            input_dim=input_dim,
            concept_dim=64,
            max_pool_size=pool_size,
            threshold=threshold,
            learning_rate=0.01,
        )

    expert_catalog = [
        ExpertConfig(name="raw-tight", pool=make_pool(input_dim=3072, threshold=0.42)),
        ExpertConfig(name="raw-loose", pool=make_pool(input_dim=3072, threshold=0.34)),
        ExpertConfig(
            name="pca-128",
            preprocessor=OnlinePCA(3072, output_dim=128, max_samples=256, seed=31),
            pool=make_pool(input_dim=128, threshold=0.35),
        ),
        ExpertConfig(
            name="pca-96",
            preprocessor=OnlinePCA(3072, output_dim=96, max_samples=256, seed=32),
            pool=make_pool(input_dim=96, threshold=0.37),
        ),
        ExpertConfig(
            name="rp-160",
            preprocessor=SparseRandomProjection(3072, output_dim=160, density=0.08, seed=33),
            pool=make_pool(input_dim=160, threshold=0.33),
        ),
        ExpertConfig(
            name="rp-128",
            preprocessor=SparseRandomProjection(3072, output_dim=128, density=0.10, seed=34),
            pool=make_pool(input_dim=128, threshold=0.36),
        ),
        ExpertConfig(
            name="rp-competitive",
            preprocessor=PreprocessingPipeline(
                [
                    ("rp", SparseRandomProjection(3072, output_dim=128, density=0.08, seed=35)),
                    ("competitive", CompetitiveLearner(128, num_neurons=96, learning_rate=0.02, seed=36)),
                ]
            ),
            pool=make_pool(input_dim=96, threshold=0.31),
        ),
        ExpertConfig(
            name="pca-competitive",
            preprocessor=PreprocessingPipeline(
                [
                    ("pca", OnlinePCA(3072, output_dim=96, max_samples=256, seed=37)),
                    ("competitive", CompetitiveLearner(96, num_neurons=96, learning_rate=0.02, seed=38)),
                ]
            ),
            pool=make_pool(input_dim=96, threshold=0.30),
        ),
        ExpertConfig(
            name="rp-192",
            preprocessor=SparseRandomProjection(3072, output_dim=192, density=0.06, seed=39),
            pool=make_pool(input_dim=192, threshold=0.38),
        ),
        ExpertConfig(
            name="pca-160",
            preprocessor=OnlinePCA(3072, output_dim=160, max_samples=256, seed=40),
            pool=make_pool(input_dim=160, threshold=0.39),
        ),
    ]

    for experts in EXPERT_COUNTS:
        print(f"[expert_count] {experts}")
        ensemble = EnsemblePool(
            EnsembleConfig(
                num_experts=experts,
                voting_method="weighted",
                abstention_threshold=0.5,
                use_boosting=True,
                diversity_target=0.3,
                expert_configs=expert_catalog[:experts],
            )
        )
        trainer = EnsembleTrainer(ensemble, num_classes=10, log_every=0)
        start = time.perf_counter()
        trainer.train(subset)
        train_seconds = time.perf_counter() - start
        accuracy, active_cccs, id_scores, ood_scores = _evaluate_ensemble(ensemble, test_samples, ood_samples)
        proxy_config = _ensemble_proxy_config(ensemble)
        results.append(
            SweepResult(
                sweep="expert_count",
                label=f"experts={experts}",
                parameter=float(experts),
                accuracy=float(accuracy),
                train_seconds=float(train_seconds),
                energy_joules=_estimate_total_energy(
                    proxy_config,
                    active_cccs=active_cccs,
                    train_samples=len(subset),
                    eval_samples=TEST_SAMPLES,
                    ood_samples=len(ood_samples),
                ),
                ood_auroc=_auroc(id_scores, ood_scores),
                active_cccs=float(active_cccs),
                source=source,
                model="ensemble",
            )
        )
    return results


def run_data_sweep(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
    source: str,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    for volume in DATA_VOLUMES:
        print(f"[data_volume] {volume}")
        subset = train_samples[:volume]
        trainer = VisionTrainer(_vision_base_config(max_pool_size=200, num_train_samples=len(subset)))
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train_online(subset, num_samples=len(subset), interleave_classes=True)
        train_seconds = time.perf_counter() - start
        metrics = trainer.evaluate(test_samples, num_samples=TEST_SAMPLES)
        active_cccs = float(metrics["mean_firing_count"])
        id_scores, ood_scores = _vision_scores(trainer, test_samples, ood_samples)
        results.append(
            SweepResult(
                sweep="data_volume",
                label=f"train={volume}",
                parameter=float(volume),
                accuracy=float(metrics["accuracy"]),
                train_seconds=float(train_seconds),
                energy_joules=_estimate_total_energy(
                    trainer.system.config,
                    active_cccs=active_cccs,
                    train_samples=len(subset),
                    eval_samples=TEST_SAMPLES,
                    ood_samples=len(ood_samples),
                ),
                ood_auroc=_auroc(id_scores, ood_scores),
                active_cccs=active_cccs,
                source=source,
                model="vision_trainer",
            )
        )
    return results


def _print_sweep(name: str, results: list[SweepResult]) -> None:
    print(f"\n=== {name} ===")
    print(_format_table(results))
    best = max(results, key=lambda result: result.efficiency)
    print(
        "findings: "
        f"diminishing_returns={_diminishing_returns(results)} | "
        f"trend={_trend_label(results)} | "
        f"cost_effective={best.label}"
    )


def _write_summary(
    path: Path,
    *,
    pool_results: list[SweepResult],
    depth_results: list[SweepResult],
    expert_results: list[SweepResult],
    data_results: list[SweepResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_results = pool_results + depth_results + expert_results + data_results
    best_overall = max(all_results, key=lambda result: result.efficiency)
    best_accuracy = max(all_results, key=lambda result: result.accuracy)
    payload = {
        "pool_size": [result.__dict__ | {"efficiency": result.efficiency} for result in pool_results],
        "hierarchy_depth": [result.__dict__ | {"efficiency": result.efficiency} for result in depth_results],
        "expert_count": [result.__dict__ | {"efficiency": result.efficiency} for result in expert_results],
        "data_volume": [result.__dict__ | {"efficiency": result.efficiency} for result in data_results],
        "findings": {
            "pool_size": {
                "diminishing_returns": _diminishing_returns(pool_results),
                "trend": _trend_label(pool_results),
                "cost_effective": max(pool_results, key=lambda result: result.efficiency).label,
            },
            "hierarchy_depth": {
                "diminishing_returns": _diminishing_returns(depth_results),
                "trend": _trend_label(depth_results),
                "cost_effective": max(depth_results, key=lambda result: result.efficiency).label,
            },
            "expert_count": {
                "diminishing_returns": _diminishing_returns(expert_results),
                "trend": _trend_label(expert_results),
                "cost_effective": max(expert_results, key=lambda result: result.efficiency).label,
            },
            "data_volume": {
                "diminishing_returns": _diminishing_returns(data_results),
                "trend": _trend_label(data_results),
                "cost_effective": max(data_results, key=lambda result: result.efficiency).label,
            },
            "best_overall": best_overall.__dict__ | {"efficiency": best_overall.efficiency},
            "best_accuracy": best_accuracy.__dict__ | {"efficiency": best_accuracy.efficiency},
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    train_samples, test_samples, source = _load_data()
    ood_samples = _make_ood_samples(OOD_SAMPLES, seed=OOD_SEED)

    print("Bio-ARN scaling-law analysis")
    print(f"data_source: {source}")
    print(
        f"default_train_samples: {DEFAULT_TRAIN_SAMPLES} | "
        f"test_samples: {TEST_SAMPLES} | ood_samples: {OOD_SAMPLES}"
    )

    pool_results = run_pool_size_sweep(train_samples, test_samples, ood_samples, source)
    depth_results = run_depth_sweep(train_samples, test_samples, ood_samples, source)
    expert_results = run_expert_sweep(train_samples, test_samples, ood_samples, source)
    data_results = run_data_sweep(train_samples, test_samples, ood_samples, source)

    _print_sweep("Pool size sweep", pool_results)
    _print_sweep("Hierarchy depth sweep", depth_results)
    _print_sweep("Expert count sweep", expert_results)
    _print_sweep("Data volume sweep", data_results)

    all_results = pool_results + depth_results + expert_results + data_results
    best_overall = max(all_results, key=lambda result: result.efficiency)
    best_accuracy = max(all_results, key=lambda result: result.accuracy)
    print("\n=== Overall summary ===")
    print(
        f"most_cost_effective: {best_overall.sweep}:{best_overall.label} "
        f"(acc={best_overall.accuracy:.3f}, energy={best_overall.energy_joules:.6f} J, "
        f"time={best_overall.train_seconds:.2f}s)"
    )
    print(
        f"best_accuracy: {best_accuracy.sweep}:{best_accuracy.label} "
        f"(acc={best_accuracy.accuracy:.3f}, ood_auroc={best_accuracy.ood_auroc:.3f})"
    )
    _write_summary(
        Path("logs") / "scaling_laws_summary.json",
        pool_results=pool_results,
        depth_results=depth_results,
        expert_results=expert_results,
        data_results=data_results,
    )


if __name__ == "__main__":
    main()
