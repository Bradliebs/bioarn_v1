"""Train a diverse Bio-ARN ensemble on synthetic CIFAR-like images."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import torch

from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool
from bioarn.training import SyntheticCIFAR10Stream, take_samples


@dataclass
class MetricRow:
    name: str
    accuracy: float
    coverage: float
    covered_accuracy: float
    abstention_rate: float


def _format_table(rows: list[MetricRow]) -> str:
    headers = ["model", "accuracy", "coverage", "covered_acc", "abstain"]
    values = [
        [
            row.name,
            f"{row.accuracy:.3f}",
            f"{row.coverage:.3f}",
            f"{row.covered_accuracy:.3f}",
            f"{row.abstention_rate:.3f}",
        ]
        for row in rows
    ]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    body = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in values]
    return "\n".join([header_line, divider, *body])


def _base_config() -> dict[str, object]:
    return {
        "input_dim": 3072,
        "concept_dim": 128,
        "max_pool_size": 200,
        "learning_rate": 0.02,
        "image_size": (32, 32, 3),
        "num_classes": 10,
    }


def _make_noisy_test_set(
    samples: list[tuple[torch.Tensor, int | None]],
    *,
    noise_scale: float = 0.18,
    seed: int = 101,
) -> list[tuple[torch.Tensor, int | None]]:
    noisy_samples: list[tuple[torch.Tensor, int | None]] = []
    generator = torch.Generator().manual_seed(seed)
    for image, label in samples:
        noisy = (image + (noise_scale * torch.randn(image.shape, generator=generator))).clamp(0.0, 1.0)
        noisy_samples.append((noisy, label))
    return noisy_samples


def _build_ensemble(*, use_boosting: bool) -> EnsemblePool:
    manager = DiversityManager()
    expert_configs = manager.create_diverse_experts(_base_config(), num_experts=4)
    return EnsemblePool(
        EnsembleConfig(
            num_experts=4,
            voting_method="weighted",
            abstention_threshold=0.5,
            use_boosting=use_boosting,
            diversity_target=0.3,
            expert_configs=expert_configs,
        )
    )


def _train_ensemble(ensemble: EnsemblePool, train_samples: list[tuple[torch.Tensor, int | None]]) -> None:
    progress_interval = max(1, len(train_samples) // 10)
    for index, (image, label) in enumerate(train_samples, start=1):
        for expert_state in ensemble.experts:
            ensemble._learn_expert(expert_state, image, label)  # noqa: SLF001
        if index % progress_interval == 0 or index == len(train_samples):
            print(f"[train] {index}/{len(train_samples)}")


def _evaluate_experts(
    ensemble: EnsemblePool,
    samples: list[tuple[torch.Tensor, int | None]],
) -> tuple[dict[str, MetricRow], list[list[int]], float]:
    totals = {expert.name: 0 for expert in ensemble.experts}
    correct = {expert.name: 0 for expert in ensemble.experts}
    covered = {expert.name: 0 for expert in ensemble.experts}
    prediction_histories = [[] for _ in ensemble.experts]
    agreements: list[float] = []

    original_voting = ensemble.config.voting_method
    original_boosting = ensemble.config.use_boosting
    ensemble.config.voting_method = "weighted"
    ensemble.config.use_boosting = False
    try:
        for image, label in samples:
            result = ensemble.classify(image)
            agreements.append(result.agreement)
            for expert_index, expert_result in enumerate(result.expert_results):
                totals[expert_result.name] += 1
                prediction_histories[expert_index].append(int(expert_result.predicted_class))
                if not expert_result.abstained:
                    covered[expert_result.name] += 1
                if label is not None and expert_result.predicted_class == label:
                    correct[expert_result.name] += 1
    finally:
        ensemble.config.voting_method = original_voting
        ensemble.config.use_boosting = original_boosting

    rows = {
        name: MetricRow(
            name=name,
            accuracy=correct[name] / max(totals[name], 1),
            coverage=covered[name] / max(totals[name], 1),
            covered_accuracy=correct[name] / max(covered[name], 1),
            abstention_rate=1.0 - (covered[name] / max(totals[name], 1)),
        )
        for name in totals
    }
    return rows, prediction_histories, float(fmean(agreements) if agreements else 0.0)


def _evaluate_ensemble(
    ensemble: EnsemblePool,
    samples: list[tuple[torch.Tensor, int | None]],
    *,
    name: str,
    voting_method: str,
    use_boosting: bool,
    update_boosting: bool = False,
) -> tuple[MetricRow, float]:
    original_voting = ensemble.config.voting_method
    original_boosting = ensemble.config.use_boosting
    ensemble.config.voting_method = voting_method
    ensemble.config.use_boosting = use_boosting

    total = 0
    correct = 0
    covered = 0
    high_agreement_correct = 0
    high_agreement_total = 0
    try:
        for image, label in samples:
            result = ensemble.classify(image)
            total += 1
            covered += int(not result.abstained)
            if label is not None and result.predicted_class == label:
                correct += 1
            if result.agreement >= 0.75:
                high_agreement_total += 1
                high_agreement_correct += int(label is not None and result.predicted_class == label)
            if update_boosting and label is not None:
                ensemble.update_with_feedback(result, int(label))
    finally:
        ensemble.config.voting_method = original_voting
        ensemble.config.use_boosting = original_boosting

    row = MetricRow(
        name=name,
        accuracy=correct / max(total, 1),
        coverage=covered / max(total, 1),
        covered_accuracy=correct / max(covered, 1),
        abstention_rate=1.0 - (covered / max(total, 1)),
    )
    high_agreement_accuracy = high_agreement_correct / max(high_agreement_total, 1)
    return row, high_agreement_accuracy


def run_experiment() -> dict[str, object]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    train_samples = take_samples(SyntheticCIFAR10Stream(3000, seed=7), 3000)
    clean_test_samples = take_samples(SyntheticCIFAR10Stream(1000, seed=8, shuffle=False), 1000)
    test_samples = _make_noisy_test_set(clean_test_samples)

    print("data_source: synthetic-cifar10")
    print("evaluation_split: synthetic-cifar10-noisy")
    print(f"train_samples: {len(train_samples)}")
    print(f"test_samples: {len(test_samples)}")

    ensemble = _build_ensemble(use_boosting=True)
    _train_ensemble(ensemble, train_samples)

    individual_rows, prediction_histories, agreement_rate = _evaluate_experts(ensemble, test_samples)
    diversity = DiversityManager().measure_diversity(prediction_histories)

    majority_row, majority_high_agreement = _evaluate_ensemble(
        ensemble,
        test_samples,
        name="ensemble-majority",
        voting_method="majority",
        use_boosting=False,
    )
    weighted_row, weighted_high_agreement = _evaluate_ensemble(
        ensemble,
        test_samples,
        name="ensemble-weighted",
        voting_method="weighted",
        use_boosting=False,
    )
    boosting_row, boosting_high_agreement = _evaluate_ensemble(
        ensemble,
        test_samples,
        name="ensemble-boosting",
        voting_method="weighted",
        use_boosting=True,
        update_boosting=True,
    )

    rows = list(individual_rows.values()) + [majority_row, weighted_row, boosting_row]
    best_single = max(individual_rows.values(), key=lambda row: (row.accuracy, row.covered_accuracy, row.coverage))
    best_ensemble = max([majority_row, weighted_row, boosting_row], key=lambda row: (row.accuracy, row.covered_accuracy, row.coverage))

    print("comparison_table:")
    print(_format_table(rows))
    print(f"best_single: {best_single.name} acc={best_single.accuracy:.3f}")
    print(f"best_ensemble: {best_ensemble.name} acc={best_ensemble.accuracy:.3f}")
    print(f"agreement_rate: {agreement_rate:.3f}")
    print(f"diversity: {diversity:.3f}")
    print(
        "high_agreement_accuracy: "
        f"majority={majority_high_agreement:.3f} "
        f"weighted={weighted_high_agreement:.3f} "
        f"boosting={boosting_high_agreement:.3f}"
    )

    return {
        "individual": individual_rows,
        "ensemble": {
            "majority": majority_row,
            "weighted": weighted_row,
            "boosting": boosting_row,
        },
        "agreement_rate": agreement_rate,
        "diversity": diversity,
        "best_single": best_single,
        "best_ensemble": best_ensemble,
    }


if __name__ == "__main__":
    run_experiment()
