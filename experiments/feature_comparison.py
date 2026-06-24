"""Compare Bio-ARN vision features across classic and learned preprocessors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from bioarn.preprocessing import (
    CompetitiveLearner,
    HebbianSparseCoder,
    OnlineDictionaryLearner,
    OnlinePCA,
    PreprocessingPipeline,
    SparseRandomProjection,
)
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples


@dataclass
class FeatureResult:
    name: str
    accuracy: float
    covered_accuracy: float
    abstention_rate: float
    sparsity: float
    reconstruction_error: float
    output_dim: int
    committed_cccs: int


def _base_config(*, num_train_samples: int, num_test_samples: int) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=256,
        margin_threshold=0.40,
        use_batched=True,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        preprocessing_warmup_samples=0,
    )


def _pipeline_factories() -> list[tuple[str, Callable[[], PreprocessingPipeline | None]]]:
    return [
        ("raw", lambda: None),
        ("pca", lambda: PreprocessingPipeline([("pca", OnlinePCA(3072, output_dim=128, max_samples=512, seed=21))])),
        (
            "random_projection",
            lambda: PreprocessingPipeline(
                [("random_projection", SparseRandomProjection(3072, output_dim=192, density=0.1, seed=22))]
            ),
        ),
        (
            "sparse_coding",
            lambda: PreprocessingPipeline(
                [("sparse", HebbianSparseCoder(3072, 512, sparsity=0.05, learning_rate=0.01, seed=23))]
            ),
        ),
        (
            "dictionary",
            lambda: PreprocessingPipeline(
                [
                    (
                        "dictionary",
                        OnlineDictionaryLearner(
                            3072,
                            dict_size=192,
                            sparsity_target=0.05,
                            learning_rate=0.03,
                            max_matching_iters=4,
                            seed=24,
                        ),
                    )
                ]
            ),
        ),
        (
            "competitive",
            lambda: PreprocessingPipeline(
                [("competitive", CompetitiveLearner(3072, num_neurons=192, learning_rate=0.03, seed=25))]
            ),
        ),
    ]


def _stack_samples(samples: list[tuple[torch.Tensor, int | None]]) -> torch.Tensor:
    return torch.stack([tensor for tensor, _ in samples], dim=0).to(torch.float32)


def _last_step(pipeline: PreprocessingPipeline | None):
    if pipeline is None:
        return None
    return pipeline.steps[-1][1]


def _prepare_pipeline(
    pipeline: PreprocessingPipeline | None,
    train_batch: torch.Tensor,
) -> PreprocessingPipeline | None:
    if pipeline is not None and not pipeline.is_fitted:
        pipeline.fit(train_batch)
    return pipeline


def _reconstruction_error(
    pipeline: PreprocessingPipeline | None,
    batch: torch.Tensor,
) -> float:
    if pipeline is None:
        return 0.0
    step = _last_step(pipeline)
    if step is None:
        return 0.0
    if hasattr(step, "reconstruction_error"):
        return float(step.reconstruction_error(batch))
    if isinstance(step, OnlinePCA):
        projected = step.transform(batch)
        reconstructions = projected @ step.components.to(projected) + step.mean.to(projected)
        return float(torch.mean((reconstructions - batch) ** 2).item())
    if isinstance(step, SparseRandomProjection):
        projected = step.transform(batch)
        forward_matrix = step.projection.to(projected).transpose(0, 1)
        reconstructions = projected @ torch.linalg.pinv(forward_matrix)
        return float(torch.mean((reconstructions - batch) ** 2).item())
    return float("nan")


def _feature_sparsity(pipeline: PreprocessingPipeline | None, batch: torch.Tensor) -> tuple[float, int]:
    if pipeline is None:
        features = batch
    else:
        features = pipeline.transform(batch)
    sparsity = float((features.abs() <= 1e-6).to(torch.float32).mean().item())
    return sparsity, int(features.shape[-1])


def _format_table(results: list[FeatureResult]) -> str:
    headers = [
        "config",
        "acc",
        "covered",
        "abstain",
        "sparsity",
        "recon_mse",
        "dim",
        "cccs",
    ]
    rows = [
        [
            result.name,
            f"{result.accuracy:.3f}",
            f"{result.covered_accuracy:.3f}",
            f"{result.abstention_rate:.3f}",
            f"{result.sparsity:.3f}",
            f"{result.reconstruction_error:.4f}",
            str(result.output_dim),
            str(result.committed_cccs),
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    body = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([header_line, divider, *body])


def _run_single(
    *,
    name: str,
    pipeline: PreprocessingPipeline | None,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    stats_batch: torch.Tensor,
) -> FeatureResult:
    config = _base_config(num_train_samples=len(train_samples), num_test_samples=len(test_samples))
    trainer = VisionTrainer(config, preprocessing=pipeline)
    trainer.train_online(train_samples, num_samples=len(train_samples))
    metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
    analysis = trainer.get_ccc_analysis()
    sparsity, output_dim = _feature_sparsity(pipeline, stats_batch)
    recon_error = _reconstruction_error(pipeline, stats_batch)
    result = FeatureResult(
        name=name,
        accuracy=float(metrics["accuracy"]),
        covered_accuracy=float(metrics["covered_accuracy"]),
        abstention_rate=float(metrics["abstention_rate"]),
        sparsity=sparsity,
        reconstruction_error=recon_error,
        output_dim=output_dim,
        committed_cccs=int(analysis["committed_cccs"]),
    )
    print(
        f"[result] {name:<18} acc={result.accuracy:.3f} "
        f"sparsity={result.sparsity:.3f} recon={result.reconstruction_error:.4f}"
    )
    return result


def main() -> None:
    train_count = 2000
    test_count = 400
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=train_count,
        test_samples=test_count,
        seed=41,
        timeout_seconds=5.0,
    )
    train_samples = take_samples(train_stream, train_count)
    test_samples = take_samples(test_stream, test_count)
    train_batch = _stack_samples(train_samples)
    stats_batch = train_batch[:128]

    print(f"data_source: {source}")
    print(f"train_samples: {len(train_samples)}")
    print(f"test_samples: {len(test_samples)}")

    results: list[FeatureResult] = []
    for name, factory in _pipeline_factories():
        print(f"[run] {name}")
        pipeline = _prepare_pipeline(factory(), train_batch)
        results.append(
            _run_single(
                name=name,
                pipeline=pipeline,
                train_samples=train_samples,
                test_samples=test_samples,
                stats_batch=stats_batch,
            )
        )

    print("feature_comparison_table:")
    print(_format_table(results))


if __name__ == "__main__":
    main()
