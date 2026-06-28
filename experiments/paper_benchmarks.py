"""Paper-ready real-data benchmarks for Bio-ARN 2.0."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import PrecisionConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.data import FashionMNISTStream, MNISTStream
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples
from real_cifar_comparison import run_both, run_hierarchy_baseline


SEED = 7
GPU_FRAME_ENERGY_MJ = 50.01
EFFICIENCY_RATIO_X = 278.0
LOIHI_FRAME_ENERGY_MJ = GPU_FRAME_ENERGY_MJ / EFFICIENCY_RATIO_X


@dataclass(frozen=True)
class BenchmarkRow:
    dataset: str
    config: str
    accuracy: float
    ood_auroc: float
    energy_mj: float
    train_seconds: float
    source: str


@dataclass(frozen=True)
class MNISTFamilySpec:
    name: str
    stream_cls: type[MNISTStream] | type[FashionMNISTStream]
    train_per_label: int = 250
    test_per_label: int = 50


@dataclass(frozen=True)
class ConceptBankConfig:
    name: str
    curiosity_weight: float = 0.0
    use_precision: bool = False
    use_locking: bool = False
    num_passes: int = 2


class ConceptPrototypeBank:
    def __init__(self, *, max_entries_per_label: int = 5, recruit_threshold: float = 0.80) -> None:
        self.max_entries_per_label = int(max_entries_per_label)
        self.recruit_threshold = float(recruit_threshold)
        self.entries: list[dict[str, int | torch.Tensor]] = []

    def observe(self, concept: torch.Tensor, label: int) -> None:
        concept = normalize(concept.reshape(1, -1)).squeeze(0)
        same_label = [(index, entry) for index, entry in enumerate(self.entries) if int(entry["label"]) == int(label)]
        if not same_label:
            self.entries.append({"label": int(label), "count": 1, "prototype": concept.clone()})
            return

        similarities = torch.tensor(
            [float(cosine_similarity(entry["prototype"], concept).item()) for _, entry in same_label],
            dtype=torch.float32,
        )
        best_local = int(torch.argmax(similarities).item())
        best_index, best_entry = same_label[best_local]
        if (
            float(similarities[best_local].item()) < self.recruit_threshold
            and len(same_label) < self.max_entries_per_label
        ):
            self.entries.append({"label": int(label), "count": 1, "prototype": concept.clone()})
            return

        previous_count = int(best_entry["count"])
        updated_count = previous_count + 1
        updated_prototype = normalize(
            ((best_entry["prototype"] * previous_count) + concept).reshape(1, -1)
        ).squeeze(0)
        self.entries[best_index] = {
            "label": int(label),
            "count": updated_count,
            "prototype": updated_prototype,
        }

    def predict(self, concept: torch.Tensor, *, threshold: float) -> tuple[int | None, float, bool]:
        if not self.entries:
            return None, 0.0, True
        concept = normalize(concept.reshape(1, -1)).squeeze(0)
        similarities = torch.tensor(
            [float(cosine_similarity(entry["prototype"], concept).item()) for entry in self.entries],
            dtype=torch.float32,
        )
        best_index = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_index].item())
        label = int(self.entries[best_index]["label"])
        return (label if best_score >= threshold else None), best_score, best_score < threshold


def _auroc(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
    positives = [(score, 1) for score in id_scores]
    negatives = [(score, 0) for score in ood_scores]
    all_points = sorted(positives + negatives, key=lambda item: item[0], reverse=True)

    total_pos = len(id_scores)
    total_neg = len(ood_scores)
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    prev_score: float | None = None
    area = 0.0

    for score, label in all_points:
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


def _energy_mj(samples_processed: int) -> float:
    return float(samples_processed) * LOIHI_FRAME_ENERGY_MJ


def _collect_balanced_samples(stream, per_label: int) -> list[tuple[torch.Tensor, int]]:
    counts: defaultdict[int, int] = defaultdict(int)
    samples: list[tuple[torch.Tensor, int]] = []
    for sample in stream.stream():
        label = int(sample.label)
        if counts[label] >= per_label:
            continue
        samples.append((sample.data, label))
        counts[label] += 1
        if all(counts[index] >= per_label for index in range(10)):
            return samples
    raise RuntimeError(f"Unable to collect {per_label} samples per class; counts={dict(counts)}")


def _build_precision_config(pool_size: int) -> PrecisionConfig:
    return PrecisionConfig(
        enabled=True,
        pool_size=pool_size,
        entropy_window=100,
        precision_alpha=5.0,
        precision_threshold=0.5,
        min_precision=0.1,
        max_precision=1.0,
    )


def _trainer_embedding(trainer: VisionTrainer, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    step = trainer._step_pool(  # noqa: SLF001
        trainer._prepare_tensor(tensor),  # noqa: SLF001
        allow_recruit=False,
        preview=True,
    )
    return step.concept_direction.detach().clone(), float(step.confidence)


def _evaluate_with_concept_bank(
    bank: ConceptPrototypeBank,
    threshold: float,
    embeddings: Sequence[tuple[torch.Tensor, int | None]],
) -> tuple[float, float, list[float]]:
    correct = 0
    abstained = 0
    confidences: list[float] = []
    for concept, label in embeddings:
        prediction, score, is_abstained = bank.predict(concept, threshold=threshold)
        confidences.append(score)
        abstained += int(is_abstained)
        correct += int(label is not None and (not is_abstained) and prediction == int(label))
    return (
        correct / max(len(embeddings), 1),
        abstained / max(len(embeddings), 1),
        confidences,
    )


def _embed_samples(
    trainer: VisionTrainer,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    return [(_trainer_embedding(trainer, tensor)[0], label) for tensor, label in samples]


def _run_mnist_family_benchmark(
    spec: MNISTFamilySpec,
    ood_stream_cls: type[MNISTStream] | type[FashionMNISTStream],
    data_dir: Path,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    train_samples = _collect_balanced_samples(
        spec.stream_cls(split="train", data_dir=str(data_dir), flatten=True, normalize=True, shuffle=True, seed=SEED),
        spec.train_per_label,
    )
    test_samples = _collect_balanced_samples(
        spec.stream_cls(split="test", data_dir=str(data_dir), flatten=True, normalize=True, shuffle=False, seed=SEED),
        spec.test_per_label,
    )
    ood_samples = _collect_balanced_samples(
        ood_stream_cls(split="test", data_dir=str(data_dir), flatten=True, normalize=True, shuffle=False, seed=SEED + 3),
        spec.test_per_label,
    )

    configs = [
        ConceptBankConfig(name="Baseline"),
        ConceptBankConfig(name="+Curiosity", curiosity_weight=0.3),
        ConceptBankConfig(name="+Precision+Lock", use_precision=True, use_locking=True),
    ]
    source = f"{spec.stream_cls.__name__} raw IDX"

    for config in configs:
        print(f"[run] {spec.name} / {config.name}")
        trainer_config = VisionTrainConfig(
            input_dim=784,
            concept_dim=128,
            max_pool_size=100,
            margin_threshold=0.50,
            use_batched=True,
            batch_size=32,
            learning_rate=0.02,
            num_train_samples=len(train_samples),
            num_test_samples=len(test_samples),
            preprocessing_warmup_samples=128,
            curiosity_weight=config.curiosity_weight,
            precision=_build_precision_config(100) if config.use_precision else None,
            num_f1_features=256,
            f1_top_k=50,
        )
        trainer = VisionTrainer(trainer_config)
        if config.use_locking:
            trainer.system.ccc_pool.config.lock_threshold = 0.8
        else:
            trainer.system.ccc_pool.config.lock_threshold = 1.1

        start = time.perf_counter()
        trainer.train_online(
            train_samples,
            num_samples=len(train_samples),
            num_passes=config.num_passes,
            interleave_classes=True,
        )

        train_embeddings = _embed_samples(trainer, train_samples)
        test_embeddings = _embed_samples(trainer, test_samples)
        ood_embeddings = _embed_samples(trainer, [(tensor, None) for tensor, _ in ood_samples])

        bank = ConceptPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)
        for concept, label in train_embeddings:
            bank.observe(concept, int(label))

        threshold = 0.55
        accuracy, _, id_scores = _evaluate_with_concept_bank(bank, threshold, test_embeddings)
        _, _, ood_scores = _evaluate_with_concept_bank(bank, threshold, ood_embeddings)
        elapsed = time.perf_counter() - start
        sample_count = (
            (len(train_samples) * config.num_passes)
            + len(train_samples)
            + len(test_samples)
            + len(ood_samples)
        )
        rows.append(
            BenchmarkRow(
                dataset=spec.name,
                config=config.name,
                accuracy=accuracy,
                ood_auroc=_auroc(id_scores, ood_scores),
                energy_mj=_energy_mj(sample_count),
                train_seconds=float(elapsed),
                source=source,
            )
        )
    return rows


def _ood_confidences(trainer: VisionTrainer, samples: Iterable[torch.Tensor]) -> list[float]:
    scores: list[float] = []
    for tensor in samples:
        step = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
            preview=True,
        )
        scores.append(float(step.confidence))
    return scores


def _evaluate_pool_samples(
    trainer: VisionTrainer,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> tuple[float, list[float]]:
    correct = 0
    confidences: list[float] = []
    for tensor, label in samples:
        step = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
            preview=True,
        )
        prediction = (
            None
            if step.abstained
            else trainer._recognition_label(step.concept_direction, step.fired_indices)  # noqa: SLF001
        )
        confidences.append(float(step.confidence))
        correct += int(label is not None and prediction == label)
    return correct / max(len(samples), 1), confidences


def _run_conv_cifar_benchmark(data_dir: Path, train_n: int, test_n: int, ood_n: int) -> BenchmarkRow:
    print("[run] CIFAR-10 / ConvCCC")
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir=str(data_dir),
        train_samples=train_n,
        test_samples=test_n,
        seed=SEED,
    )
    if source != "cifar10":
        raise RuntimeError(f"Expected real CIFAR-10 data, got fallback source: {source}")

    train_samples = take_samples(train_stream, train_n)
    test_samples = take_samples(test_stream, test_n)
    generator = torch.Generator().manual_seed(SEED + 99)
    ood_samples = [torch.rand(3072, generator=generator, dtype=torch.float32) for _ in range(ood_n)]

    trainer = VisionTrainer(
        VisionTrainConfig(
            input_dim=3072,
            concept_dim=256,
            max_pool_size=100,
            margin_threshold=0.35,
            use_batched=False,
            batch_size=32,
            learning_rate=0.01,
            num_train_samples=len(train_samples),
            num_test_samples=len(test_samples),
            preprocessing_warmup_samples=0,
            use_conv_ccc=True,
        )
    )
    trainer.system.ccc_pool.config.lock_threshold = 1.1

    start = time.perf_counter()
    trainer.train_online(
        train_samples,
        num_samples=len(train_samples),
        num_passes=3,
        interleave_classes=True,
    )
    accuracy, id_scores = _evaluate_pool_samples(trainer, test_samples)
    ood_scores = _ood_confidences(trainer, ood_samples)
    elapsed = time.perf_counter() - start
    return BenchmarkRow(
        dataset="CIFAR-10",
        config="ConvCCC (3-pass)",
        accuracy=float(accuracy),
        ood_auroc=_auroc(id_scores, ood_scores),
        energy_mj=_energy_mj((len(train_samples) * 3) + len(test_samples) + len(ood_samples)),
        train_seconds=float(elapsed),
        source="CIFAR10Stream real batches",
    )


def _run_hierarchy_cifar_benchmarks(data_dir: Path, train_n: int, test_n: int, ood_n: int) -> list[BenchmarkRow]:
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir=str(data_dir),
        train_samples=train_n,
        test_samples=test_n,
        seed=SEED,
    )
    if source != "cifar10":
        raise RuntimeError(f"Expected real CIFAR-10 data, got fallback source: {source}")

    train_samples = take_samples(train_stream, train_n)
    test_samples = take_samples(test_stream, test_n)
    generator = torch.Generator().manual_seed(SEED + 42)
    ood_samples = [torch.rand(3072, generator=generator, dtype=torch.float32) for _ in range(ood_n)]

    rows: list[BenchmarkRow] = []
    for config_name, runner, train_factor, extra_events in (
        ("Hierarchy", run_hierarchy_baseline, 1, 0),
        ("+Hierarchy+Ens", run_both, 2, max(1, train_n // 6) + min(100, ood_n)),
    ):
        print(f"[run] CIFAR-10 / {config_name}")
        start = time.perf_counter()
        result = runner(train_samples, test_samples, ood_samples)
        elapsed = time.perf_counter() - start
        rows.append(
            BenchmarkRow(
                dataset="CIFAR-10",
                config=config_name,
                accuracy=float(result.accuracy),
                ood_auroc=float(result.ood_auroc),
                energy_mj=_energy_mj((train_n * train_factor) + test_n + ood_n + extra_events),
                train_seconds=float(elapsed),
                source="CIFAR10Stream real batches",
            )
        )
    return rows


def _print_table(rows: Sequence[BenchmarkRow]) -> None:
    title = "Bio-ARN 2.0 — Paper-Ready Benchmarks"
    headers = ("Dataset", "Config", "Accuracy", "OOD AUROC", "Energy", "Time")
    body = [
        (
            row.dataset,
            row.config,
            f"{row.accuracy * 100:5.1f}%",
            f"{row.ood_auroc:7.3f}",
            f"{row.energy_mj:6.1f} mJ",
            f"{row.train_seconds:6.1f}s",
        )
        for row in rows
    ]
    widths = [max(len(headers[index]), *(len(item[index]) for item in body)) for index in range(len(headers))]
    total_width = sum(widths) + (3 * (len(headers) - 1))

    print("╔" + "═" * total_width + "╗")
    print("║" + title.center(total_width) + "║")
    print("╠" + "═" * total_width + "╣")
    print("║ " + " │ ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " ║")
    print("╠" + "═" * total_width + "╣")
    for item in body:
        print("║ " + " │ ".join(item[index].ljust(widths[index]) for index in range(len(headers))) + " ║")
    print("╚" + "═" * total_width + "╝")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready real-data Bio-ARN benchmarks.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--cifar-train", type=int, default=2000)
    parser.add_argument("--cifar-test", type=int, default=200)
    parser.add_argument("--cifar-ood", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    data_dir = Path(args.data_dir)
    if not (data_dir / "cifar-10-batches-py").exists():
        raise FileNotFoundError(f"Expected real CIFAR-10 batches at {data_dir / 'cifar-10-batches-py'}")

    mnist_rows = _run_mnist_family_benchmark(
        MNISTFamilySpec(name="MNIST", stream_cls=MNISTStream),
        FashionMNISTStream,
        data_dir,
    )
    fashion_rows = _run_mnist_family_benchmark(
        MNISTFamilySpec(name="Fashion-MNIST", stream_cls=FashionMNISTStream),
        MNISTStream,
        data_dir,
    )
    cifar_conv_row = _run_conv_cifar_benchmark(data_dir, args.cifar_train, args.cifar_test, args.cifar_ood)
    cifar_rows = _run_hierarchy_cifar_benchmarks(data_dir, args.cifar_train, args.cifar_test, args.cifar_ood)

    rows = [*mnist_rows, *fashion_rows, cifar_conv_row, *cifar_rows]
    print()
    _print_table(rows)
    print()
    print("Sources:")
    for row in rows:
        print(f"- {row.dataset:<13} {row.config:<18} {row.source}")
    print(f"- Energy uses projected Loihi 2 frame energy = {LOIHI_FRAME_ENERGY_MJ:.3f} mJ ({EFFICIENCY_RATIO_X:.0f}× vs 50.01 mJ GPU baseline).")


if __name__ == "__main__":
    main()
