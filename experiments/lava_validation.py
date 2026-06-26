"""Validate the Loihi 2 export path with a portable LIF simulator."""

from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.core.math_utils import normalize
from bioarn.export import NeuromorphicGraph, export_ccc_pool
from bioarn.scaling import BatchedCCCPool
from bioarn.training import VisionTrainConfig, VisionTrainer

TRAIN_PER_CLASS = 40
TEST_PER_CLASS = 16
NUM_CLASSES = 4
IMAGE_SIZE = 8
SIM_STEPS = 24
SEED = 23


@dataclass
class ExportedCCC:
    index: int
    f1_weights: torch.Tensor
    f1_bias: torch.Tensor
    f2_weights: torch.Tensor
    feedback_weights: torch.Tensor
    concept_direction: torch.Tensor
    theta_margin: float


@dataclass
class WeightFidelity:
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float
    norm_ratio: float


@dataclass
class SpikeFidelity:
    f1_rate_correlation: float
    gate_rate_correlation: float
    fired_agreement: float
    mean_original_confidence: float
    mean_gate_rate: float


@dataclass
class AccuracyComparison:
    original_accuracy: float
    simulated_accuracy: float
    absolute_delta: float


@dataclass
class CCCSimulation:
    f1_rates: torch.Tensor
    f2_rates: torch.Tensor
    gate_rate: float


@dataclass
class ValidationSummary:
    backend: str
    committed_cccs: int
    weight_fidelity: WeightFidelity
    spike_fidelity: SpikeFidelity
    accuracy: AccuracyComparison
    export_path: str


def _make_base_patterns() -> list[torch.Tensor]:
    patterns: list[torch.Tensor] = []

    vertical = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    vertical[:, 2:4] = 1.0
    vertical[1:7, 4:5] = 0.7
    patterns.append(vertical)

    horizontal = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    horizontal[2:4, :] = 1.0
    horizontal[4:5, 1:7] = 0.7
    patterns.append(horizontal)

    diagonal = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    diagonal.fill_diagonal_(1.0)
    diagonal[:, 1:].diagonal().copy_(torch.full((IMAGE_SIZE - 1,), 0.6))
    patterns.append(diagonal)

    box = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    box[1:7, 1] = 1.0
    box[1:7, 6] = 1.0
    box[1, 1:7] = 1.0
    box[6, 1:7] = 1.0
    box[3:5, 3:5] = 0.4
    patterns.append(box)

    return patterns


def _make_symbol(label: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pattern = _make_base_patterns()[int(label)].clone()
    shift_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
    shift_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
    pattern = torch.roll(pattern, shifts=(shift_y, shift_x), dims=(0, 1))
    noise = torch.randn(pattern.shape, generator=generator) * 0.04
    gain = 0.9 + (0.2 * torch.rand(1, generator=generator).item())
    return (pattern * gain + noise).clamp_(0.0, 1.0).reshape(-1)


def _build_dataset() -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    train: list[tuple[torch.Tensor, int]] = []
    test: list[tuple[torch.Tensor, int]] = []
    for label in range(NUM_CLASSES):
        for index in range(TRAIN_PER_CLASS):
            train.append((_make_symbol(label, seed=(SEED * 100) + (label * 1000) + index), label))
        for index in range(TEST_PER_CLASS):
            test.append((_make_symbol(label, seed=(SEED * 1000) + (label * 2000) + index), label))

    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED)).tolist()
    train = [train[index] for index in order]
    return train, test


def _trainer_config(train_count: int, test_count: int) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=IMAGE_SIZE * IMAGE_SIZE,
        concept_dim=24,
        max_pool_size=32,
        margin_threshold=0.40,
        use_batched=True,
        batch_size=16,
        learning_rate=0.03,
        num_train_samples=train_count,
        num_test_samples=test_count,
        preprocessing_warmup_samples=0,
    )


def _train_small_model(
    train_samples: list[tuple[torch.Tensor, int]],
    test_samples: list[tuple[torch.Tensor, int]],
) -> tuple[VisionTrainer, dict[str, object]]:
    trainer = VisionTrainer(_trainer_config(len(train_samples), len(test_samples)))
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_online(train_samples, num_samples=len(train_samples), interleave_classes=True)
    metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
    return trainer, metrics


def _load_graph(export_path: Path) -> NeuromorphicGraph:
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    graph = NeuromorphicGraph.from_dict(payload["graph"])
    graph.validate()
    return graph


def _parse_exported_cccs(graph: NeuromorphicGraph) -> list[ExportedCCC]:
    populations = {population.id: population for population in graph.populations}
    projections = {projection.id: projection for projection in graph.projections}
    exported: list[ExportedCCC] = []

    for population_id, population in populations.items():
        if not population_id.endswith("_gate") or "ccc_pool_ccc_" not in population_id:
            continue
        index = int(population.metadata["ccc_index"])
        prefix = f"ccc_pool_ccc_{index}"
        f1_projection = projections[f"ccc_pool_input_to_{prefix}_f1"]
        f2_projection = projections[f"{prefix}_f1_to_{prefix}_f2"]
        feedback_projection = projections[f"{prefix}_f2_to_{prefix}_f1"]
        gate_projection = projections[f"{prefix}_f2_to_{prefix}_gate"]
        exported.append(
            ExportedCCC(
                index=index,
                f1_weights=torch.tensor(f1_projection.weights, dtype=torch.float32),
                f1_bias=torch.tensor(f1_projection.bias, dtype=torch.float32),
                f2_weights=torch.tensor(f2_projection.weights, dtype=torch.float32),
                feedback_weights=torch.tensor(feedback_projection.weights, dtype=torch.float32),
                concept_direction=torch.tensor(gate_projection.weights[0], dtype=torch.float32),
                theta_margin=float(population.parameters["threshold"]),
            )
        )

    exported.sort(key=lambda ccc: ccc.index)
    return exported


def _pool_committed_indices(pool: BatchedCCCPool) -> list[int]:
    return [int(index) for index in pool.committed_mask.nonzero(as_tuple=False).reshape(-1).tolist()]


def _weight_fidelity(pool: BatchedCCCPool, exported_cccs: list[ExportedCCC]) -> WeightFidelity:
    max_abs_diff = 0.0
    mean_abs_diff_sum = 0.0
    cosine_sum = 0.0
    norm_ratio_sum = 0.0
    component_count = 0

    for exported in exported_cccs:
        original_components = [
            pool.f1_weights[exported.index],
            pool.f1_bias[exported.index],
            pool.f2_weights[exported.index],
            pool.feedback_weights[exported.index],
            pool.concept_directions[exported.index],
        ]
        exported_components = [
            exported.f1_weights,
            exported.f1_bias,
            exported.f2_weights,
            exported.feedback_weights,
            exported.concept_direction,
        ]
        for original, restored in zip(original_components, exported_components, strict=False):
            diff = (original.detach().cpu() - restored.detach().cpu()).abs()
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            mean_abs_diff_sum += float(diff.mean().item())
            flat_original = original.reshape(-1).to(torch.float32)
            flat_restored = restored.reshape(-1).to(torch.float32)
            cosine_sum += float(F.cosine_similarity(flat_original.unsqueeze(0), flat_restored.unsqueeze(0)).item())
            norm_ratio_sum += float(flat_restored.norm().item() / max(flat_original.norm().item(), 1e-8))
            component_count += 1

    return WeightFidelity(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff_sum / max(component_count, 1),
        cosine_similarity=cosine_sum / max(component_count, 1),
        norm_ratio=norm_ratio_sum / max(component_count, 1),
    )


def _original_ccc_response(pool: BatchedCCCPool, index: int, tensor: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
    raw_batch = tensor.unsqueeze(0).to(torch.float32)
    f1 = pool._row_f1_encode(index, raw_batch).squeeze(0)  # noqa: SLF001
    f2 = pool._row_f2_activate(index, f1.unsqueeze(0)).squeeze(0)  # noqa: SLF001
    confidence = float(F.cosine_similarity(f2.unsqueeze(0), pool.concept_directions[index].unsqueeze(0)).item())
    fired = confidence > float(pool.theta_margin[index].item())
    return f1, confidence, fired


def _lif_step(
    membrane: torch.Tensor,
    refractory: torch.Tensor,
    current: torch.Tensor,
    *,
    decay: float,
    threshold: float,
    reset: float,
    refractory_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = refractory <= 0
    membrane = (membrane * decay) + (current * active.to(current.dtype))
    spikes = membrane >= threshold
    membrane = torch.where(spikes, torch.full_like(membrane, reset), membrane)
    refractory = torch.where(
        spikes,
        torch.full_like(refractory, int(refractory_steps)),
        torch.clamp(refractory - 1, min=0),
    )
    return spikes.to(torch.float32), membrane, refractory


def _simulate_exported_ccc(exported: ExportedCCC, tensor: torch.Tensor) -> CCCSimulation:
    f1_mem = torch.zeros(exported.f1_weights.shape[0], dtype=torch.float32)
    f2_mem = torch.zeros(exported.f2_weights.shape[0], dtype=torch.float32)
    gate_mem = torch.zeros(1, dtype=torch.float32)
    f1_ref = torch.zeros_like(f1_mem, dtype=torch.long)
    f2_ref = torch.zeros_like(f2_mem, dtype=torch.long)
    gate_ref = torch.zeros(1, dtype=torch.long)
    f1_counts = torch.zeros_like(f1_mem)
    f2_counts = torch.zeros_like(f2_mem)
    gate_counts = torch.zeros(1, dtype=torch.float32)
    feedback_drive = torch.zeros(exported.f1_weights.shape[0], dtype=torch.float32)

    base_f1_current = torch.relu(exported.f1_weights @ tensor + exported.f1_bias)
    for _ in range(SIM_STEPS):
        f1_current = base_f1_current + (0.5 * feedback_drive)
        f1_spikes, f1_mem, f1_ref = _lif_step(
            f1_mem,
            f1_ref,
            f1_current,
            decay=0.9,
            threshold=1.0,
            reset=0.0,
            refractory_steps=1,
        )
        f2_current = exported.f2_weights @ f1_spikes
        f2_spikes, f2_mem, f2_ref = _lif_step(
            f2_mem,
            f2_ref,
            f2_current,
            decay=0.9,
            threshold=1.0,
            reset=0.0,
            refractory_steps=1,
        )
        gate_current = torch.tensor([float((exported.concept_direction * f2_spikes).sum().item())], dtype=torch.float32)
        gate_spikes, gate_mem, gate_ref = _lif_step(
            gate_mem,
            gate_ref,
            gate_current,
            decay=0.9,
            threshold=max(exported.theta_margin, 0.05),
            reset=0.0,
            refractory_steps=1,
        )
        feedback_drive = exported.feedback_weights @ f2_spikes
        f1_counts += f1_spikes
        f2_counts += f2_spikes
        gate_counts += gate_spikes

    return CCCSimulation(
        f1_rates=f1_counts / SIM_STEPS,
        f2_rates=f2_counts / SIM_STEPS,
        gate_rate=float(gate_counts.item() / SIM_STEPS),
    )


def _prototype_predict(prototypes: dict[int, torch.Tensor], concept: torch.Tensor) -> int | None:
    if not prototypes or float(concept.norm().item()) <= 1e-8:
        return None
    labels = sorted(prototypes)
    stacked = torch.stack([prototypes[label].to(concept) for label in labels], dim=0)
    query = normalize(concept.reshape(1, -1)).expand_as(stacked)
    similarities = torch.sum(stacked * query, dim=-1)
    return int(labels[int(torch.argmax(similarities).item())])


def _simulate_classification(
    exported_cccs: list[ExportedCCC],
    tensor: torch.Tensor,
    *,
    ccc_label_counts: defaultdict[int, Counter[int]],
    label_prototypes: dict[int, torch.Tensor],
) -> tuple[int, float]:
    gate_rates: dict[int, float] = {}
    fired_indices: list[int] = []
    concept_parts: list[torch.Tensor] = []

    for exported in exported_cccs:
        simulation = _simulate_exported_ccc(exported, tensor)
        gate_rates[exported.index] = simulation.gate_rate
        if simulation.gate_rate > 0.0:
            fired_indices.append(exported.index)
            concept_parts.append(exported.concept_direction * simulation.gate_rate)

    if not fired_indices:
        return -1, 0.0

    votes: defaultdict[int, float] = defaultdict(float)
    for index in fired_indices:
        gate_rate = gate_rates[index]
        counts = ccc_label_counts.get(index)
        if not counts:
            continue
        label, count = counts.most_common(1)[0]
        purity = count / max(sum(counts.values()), 1)
        votes[int(label)] += purity * gate_rate

    if votes:
        label, score = max(votes.items(), key=lambda item: item[1])
        return int(label), float(score)

    concept = normalize(torch.stack(concept_parts, dim=0).sum(dim=0, keepdim=True)).squeeze(0)
    prototype_label = _prototype_predict(label_prototypes, concept)
    return (-1, 0.0) if prototype_label is None else (int(prototype_label), max(gate_rates.values()))


def _spike_fidelity(
    pool: BatchedCCCPool,
    exported_cccs: list[ExportedCCC],
    test_samples: list[tuple[torch.Tensor, int]],
) -> SpikeFidelity:
    original_f1_values: list[float] = []
    simulated_f1_rates: list[float] = []
    original_confidences: list[float] = []
    simulated_gate_rates: list[float] = []
    fired_matches = 0
    fired_total = 0

    for tensor, _ in test_samples:
        for exported in exported_cccs:
            original_f1, confidence, fired = _original_ccc_response(pool, exported.index, tensor)
            simulated = _simulate_exported_ccc(exported, tensor)
            original_f1_values.append(float(original_f1.mean().item()))
            simulated_f1_rates.append(float(simulated.f1_rates.mean().item()))
            original_confidences.append(float(confidence))
            simulated_gate_rates.append(float(simulated.gate_rate))
            fired_matches += int(fired == (simulated.gate_rate > 0.0))
            fired_total += 1

    original_f1_tensor = torch.tensor(original_f1_values, dtype=torch.float32)
    simulated_f1_tensor = torch.tensor(simulated_f1_rates, dtype=torch.float32)
    original_conf_tensor = torch.tensor(original_confidences, dtype=torch.float32)
    simulated_gate_tensor = torch.tensor(simulated_gate_rates, dtype=torch.float32)
    return SpikeFidelity(
        f1_rate_correlation=float(F.cosine_similarity(original_f1_tensor.unsqueeze(0), simulated_f1_tensor.unsqueeze(0)).item()),
        gate_rate_correlation=float(F.cosine_similarity(original_conf_tensor.unsqueeze(0), simulated_gate_tensor.unsqueeze(0)).item()),
        fired_agreement=float(fired_matches / max(fired_total, 1)),
        mean_original_confidence=float(original_conf_tensor.mean().item()),
        mean_gate_rate=float(simulated_gate_tensor.mean().item()),
    )


def _accuracy_comparison(
    trainer: VisionTrainer,
    exported_cccs: list[ExportedCCC],
    test_samples: list[tuple[torch.Tensor, int]],
    original_metrics: dict[str, object],
) -> AccuracyComparison:
    correct = 0
    for tensor, label in test_samples:
        prediction, _ = _simulate_classification(
            exported_cccs,
            tensor,
            ccc_label_counts=trainer.ccc_label_counts,
            label_prototypes=trainer.label_bank.prototypes,
        )
        correct += int(prediction == label)
    simulated_accuracy = correct / max(len(test_samples), 1)
    original_accuracy = float(original_metrics["accuracy"])
    return AccuracyComparison(
        original_accuracy=original_accuracy,
        simulated_accuracy=float(simulated_accuracy),
        absolute_delta=abs(original_accuracy - simulated_accuracy),
    )


def _detect_lava_backend() -> str:
    try:
        import importlib.util

        if importlib.util.find_spec("lava") is None:
            return "python-lif-fallback"
        return "lava-detected-python-lif-fallback"
    except Exception:
        return "python-lif-fallback"


def _write_summary(path: Path, summary: ValidationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "backend": summary.backend,
                "committed_cccs": summary.committed_cccs,
                "weight_fidelity": summary.weight_fidelity.__dict__,
                "spike_fidelity": summary.spike_fidelity.__dict__,
                "accuracy": summary.accuracy.__dict__,
                "export_path": summary.export_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    torch.manual_seed(SEED)
    train_samples, test_samples = _build_dataset()
    trainer, original_metrics = _train_small_model(train_samples, test_samples)
    pool = trainer.system.ccc_pool
    if not isinstance(pool, BatchedCCCPool):
        raise TypeError("lava_validation expects the VisionTrainer to use BatchedCCCPool.")

    export_dir = Path("logs") / "lava_validation"
    export_graph = export_ccc_pool(pool, export_dir)
    export_path = export_dir / "ccc_pool.loihi2.json"
    graph = _load_graph(export_path)
    graph.validate()
    export_graph.validate()

    exported_cccs = _parse_exported_cccs(graph)
    committed_indices = _pool_committed_indices(pool)
    if len(exported_cccs) != len(committed_indices):
        raise ValueError("Exported committed CCC count does not match the trained pool.")

    backend = _detect_lava_backend()
    weight_fidelity = _weight_fidelity(pool, exported_cccs)
    spike_fidelity = _spike_fidelity(pool, exported_cccs, test_samples)
    accuracy = _accuracy_comparison(trainer, exported_cccs, test_samples, original_metrics)

    summary = ValidationSummary(
        backend=backend,
        committed_cccs=len(exported_cccs),
        weight_fidelity=weight_fidelity,
        spike_fidelity=spike_fidelity,
        accuracy=accuracy,
        export_path=str(export_path),
    )
    _write_summary(Path("logs") / "lava_validation_summary.json", summary)

    print("Lava export validation")
    print(f"backend: {summary.backend}")
    print(f"committed_cccs: {summary.committed_cccs}")
    print(
        "weight_fidelity: "
        f"max_abs_diff={summary.weight_fidelity.max_abs_diff:.6f} "
        f"mean_abs_diff={summary.weight_fidelity.mean_abs_diff:.6f} "
        f"cosine={summary.weight_fidelity.cosine_similarity:.6f} "
        f"norm_ratio={summary.weight_fidelity.norm_ratio:.6f}"
    )
    print(
        "spike_fidelity: "
        f"f1_corr={summary.spike_fidelity.f1_rate_correlation:.3f} "
        f"gate_corr={summary.spike_fidelity.gate_rate_correlation:.3f} "
        f"fired_agreement={summary.spike_fidelity.fired_agreement:.3f} "
        f"mean_conf={summary.spike_fidelity.mean_original_confidence:.3f} "
        f"mean_gate_rate={summary.spike_fidelity.mean_gate_rate:.3f}"
    )
    print(
        "accuracy: "
        f"original={summary.accuracy.original_accuracy:.3f} "
        f"simulated={summary.accuracy.simulated_accuracy:.3f} "
        f"delta={summary.accuracy.absolute_delta:.3f}"
    )
    print(f"export_path: {summary.export_path}")


if __name__ == "__main__":
    main()
