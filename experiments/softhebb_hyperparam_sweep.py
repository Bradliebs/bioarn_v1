"""Two-phase SoftHebb hyperparameter sweep for ConvF1Layer on real CIFAR-10."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
import warnings

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig
from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.core.math_utils import normalize

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for softhebb_hyperparam_sweep.py") from exc

PHASE1_GAMMAS = (1.0, 2.0, 4.0, 8.0, 16.0)
PHASE1_BETAS = (1.0, 1.5, 2.0, 3.0)
PHASE2_LRS = (0.001, 0.005, 0.01, 0.02)
PHASE2_THETA_DECAYS = (0.95, 0.99, 0.999)
DEFAULT_PHASE1_LR = 0.005
DEFAULT_PHASE1_THETA_DECAY = 0.99
NUM_CLASSES = 10
_WORKER_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class SweepResult:
    gamma: float
    beta: float
    lr: float
    theta_decay: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help="Balanced training subset size (must be divisible by 10).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=500,
        help="Balanced test subset size (must be divisible by 10).",
    )
    parser.add_argument("--passes", type=int, default=3, help="Training passes per configuration.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Mini-batch size.")
    parser.add_argument(
        "--prototype-samples",
        type=int,
        default=1000,
        help="Balanced subset size used to build nearest-centroid prototypes.",
    )
    parser.add_argument("--phase1-top-k", type=int, default=3, help="How many phase-1 configs to fine-tune.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Optional torch CPU thread cap. Leave at 0 to keep the default.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel worker count for CPU sweeps. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker. Leave at 0 to auto-balance across workers.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive.")
    if args.train_samples % NUM_CLASSES != 0 or args.test_samples % NUM_CLASSES != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    if args.passes <= 0:
        raise ValueError("passes must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.prototype_samples <= 0:
        raise ValueError("prototype_samples must be positive.")
    if args.prototype_samples % NUM_CLASSES != 0:
        raise ValueError("prototype_samples must be divisible by 10 for balanced sampling.")
    if args.prototype_samples > args.train_samples:
        raise ValueError("prototype_samples cannot exceed train_samples.")
    if args.phase1_top_k <= 0:
        raise ValueError("phase1_top_k must be positive.")
    if args.num_threads < 0:
        raise ValueError("num_threads cannot be negative.")
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if args.worker_threads < 0:
        raise ValueError("worker_threads cannot be negative.")


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_balanced_samples(dataset, *, per_label: int, seed: int) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    counts: defaultdict[int, int] = defaultdict(int)
    samples: list[tuple[torch.Tensor, int]] = []
    for index in order:
        image, label = dataset[index]
        label = int(label)
        if counts[label] >= per_label:
            continue
        samples.append((image.to(torch.float32), label))
        counts[label] += 1
        if all(counts[class_id] >= per_label for class_id in range(NUM_CLASSES)):
            break
    expected = per_label * NUM_CLASSES
    if len(samples) != expected:
        raise RuntimeError(f"Unable to collect {expected} balanced CIFAR-10 samples.")
    return samples


def _load_real_cifar10(
    *,
    data_dir: Path,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    train_subset = _collect_balanced_samples(train_dataset, per_label=train_samples // NUM_CLASSES, seed=seed)
    test_subset = _collect_balanced_samples(test_dataset, per_label=test_samples // NUM_CLASSES, seed=seed + 1)
    train_images = torch.stack([image for image, _ in train_subset], dim=0).to(torch.float32)
    train_labels = torch.tensor([label for _, label in train_subset], dtype=torch.long)
    test_images = torch.stack([image for image, _ in test_subset], dim=0).to(torch.float32)
    test_labels = torch.tensor([label for _, label in test_subset], dtype=torch.long)
    return train_images, train_labels, test_images, test_labels


def _balanced_tensor_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    total_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_samples >= images.shape[0]:
        return images, labels
    per_label = total_samples // NUM_CLASSES
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(labels.shape[0], generator=generator).tolist()
    counts: defaultdict[int, int] = defaultdict(int)
    selected_indices: list[int] = []
    for index in order:
        label = int(labels[index].item())
        if counts[label] >= per_label:
            continue
        selected_indices.append(index)
        counts[label] += 1
        if len(selected_indices) == total_samples:
            break
    if len(selected_indices) != total_samples:
        raise RuntimeError(f"Unable to collect {total_samples} balanced prototype samples.")
    subset_indices = torch.tensor(selected_indices, dtype=torch.long)
    return images.index_select(0, subset_indices), labels.index_select(0, subset_indices)


def _iter_image_batches(
    images: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(images.shape[0], generator=generator)
    else:
        indices = torch.arange(images.shape[0])
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images.index_select(0, batch_indices)


def _iter_labeled_batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
):
    for start in range(0, images.shape[0], batch_size):
        end = start + batch_size
        yield images[start:end], labels[start:end]


def _build_config(*, gamma: float, beta: float, lr: float, theta_decay: float, batch_size: int) -> ConvCCCConfig:
    return ConvCCCConfig(
        spatial_size=32,
        num_conv_features=64,
        f1_top_k=32,
        conv_hebbian_lr=lr,
        hebbian_batch_size=batch_size,
        softhebb_enabled=True,
        softhebb_gamma=gamma,
        softhebb_beta=beta,
        softhebb_theta_decay=theta_decay,
    )


def _build_layer(config: ConvCCCConfig, device: torch.device) -> ConvF1Layer:
    return ConvF1Layer(
        in_channels=config.in_channels,
        num_features=config.num_conv_features,
        spatial_size=config.spatial_size,
        top_k=config.f1_top_k,
        spatial_grid=config.spatial_grid,
        num_layers=config.num_conv_layers,
        hidden_channels=config.conv_hidden_channels,
        kernel_sizes=config.conv_kernel_sizes,
        spatial_top_k=config.spatial_top_k,
        competitive_k=config.conv_competitive_k,
        hebbian_lr=config.conv_hebbian_lr,
        hebbian_batch_size=config.hebbian_batch_size,
        weight_norm_target=config.conv_weight_norm,
        enable_local_contrast_norm=config.enable_local_contrast_norm,
        contrast_kernel_size=config.contrast_kernel_size,
        response_norm_eps=config.response_norm_eps,
        feature_pool_avg_mix=config.feature_pool_avg_mix,
        hebbian_oja_decay=config.hebbian_oja_decay,
        filter_decorrelation=config.filter_decorrelation,
        softhebb_enabled=config.softhebb_enabled,
        softhebb_gamma=config.softhebb_gamma,
        softhebb_beta=config.softhebb_beta,
        softhebb_theta_decay=config.softhebb_theta_decay,
    ).to(device)


@torch.inference_mode()
def _train_layer(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    *,
    passes: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    for pass_index in range(passes):
        for batch in _iter_image_batches(
            train_images,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + pass_index,
        ):
            batch = batch.to(device=device, non_blocking=True)
            learning_signal = torch.ones(batch.shape[0], device=device, dtype=torch.float32)
            layer.hebbian_update(batch, learning_signal=learning_signal)
        layer.flush_hebbian_updates()


@torch.inference_mode()
def _compute_prototypes(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    sums = torch.zeros((NUM_CLASSES, layer.output_dim), device=device, dtype=torch.float32)
    counts = torch.zeros(NUM_CLASSES, device=device, dtype=torch.float32)
    for batch_images, batch_labels in _iter_labeled_batches(train_images, train_labels, batch_size=batch_size):
        batch_images = batch_images.to(device=device, non_blocking=True)
        batch_labels = batch_labels.to(device=device, non_blocking=True)
        features = layer(batch_images).to(torch.float32)
        sums.index_add_(0, batch_labels, features)
        counts.index_add_(0, batch_labels, torch.ones_like(batch_labels, dtype=torch.float32))
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
    return normalize(prototypes)


@torch.inference_mode()
def _evaluate(
    layer: ConvF1Layer,
    prototypes: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    for batch_images, batch_labels in _iter_labeled_batches(test_images, test_labels, batch_size=batch_size):
        batch_images = batch_images.to(device=device, non_blocking=True)
        features = normalize(layer(batch_images).to(torch.float32))
        scores = features @ prototypes.T
        predictions = scores.argmax(dim=1).cpu()
        correct += int((predictions == batch_labels).sum().item())
        total += int(batch_labels.numel())
    return correct / max(total, 1)


def _evaluate_config(
    *,
    gamma: float,
    beta: float,
    lr: float,
    theta_decay: float,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    prototype_images: torch.Tensor,
    prototype_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    passes: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> SweepResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = _build_config(
        gamma=gamma,
        beta=beta,
        lr=lr,
        theta_decay=theta_decay,
        batch_size=batch_size,
    )
    layer = _build_layer(config, device)
    _train_layer(
        layer,
        train_images,
        passes=passes,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    prototypes = _compute_prototypes(
        layer,
        prototype_images,
        prototype_labels,
        batch_size=batch_size,
        device=device,
    )
    accuracy = _evaluate(
        layer,
        prototypes,
        test_images,
        test_labels,
        batch_size=batch_size,
        device=device,
    )
    return SweepResult(
        gamma=gamma,
        beta=beta,
        lr=lr,
        theta_decay=theta_decay,
        accuracy=accuracy,
    )


def _init_worker(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    prototype_images: torch.Tensor,
    prototype_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    passes: int,
    batch_size: int,
    seed: int,
    device_name: str,
    worker_threads: int,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "train_images": train_images,
        "train_labels": train_labels,
        "prototype_images": prototype_images,
        "prototype_labels": prototype_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "passes": passes,
        "batch_size": batch_size,
        "seed": seed,
        "device": torch.device(device_name),
    }
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    )


def _evaluate_worker(job: tuple[float, float, float, float]) -> tuple[SweepResult, float]:
    gamma, beta, lr, theta_decay = job
    started = time.perf_counter()
    result = _evaluate_config(
        gamma=gamma,
        beta=beta,
        lr=lr,
        theta_decay=theta_decay,
        train_images=_WORKER_STATE["train_images"],
        train_labels=_WORKER_STATE["train_labels"],
        prototype_images=_WORKER_STATE["prototype_images"],
        prototype_labels=_WORKER_STATE["prototype_labels"],
        test_images=_WORKER_STATE["test_images"],
        test_labels=_WORKER_STATE["test_labels"],
        passes=int(_WORKER_STATE["passes"]),
        batch_size=int(_WORKER_STATE["batch_size"]),
        seed=int(_WORKER_STATE["seed"]),
        device=_WORKER_STATE["device"],
    )
    return result, time.perf_counter() - started


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    divider = ["-" * max(len(header), 3) for header in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _print_phase1_summary(results: list[SweepResult], top_results: list[SweepResult]) -> None:
    print(f"=== Phase 1: Gamma × Beta Sweep (LR={DEFAULT_PHASE1_LR}, θ_decay={DEFAULT_PHASE1_THETA_DECAY}) ===\n")
    ordered_results = sorted(results, key=lambda result: (result.gamma, result.beta))
    rows = [
        [f"{result.gamma:.1f}", f"{result.beta:.1f}", _format_accuracy(result.accuracy)]
        for result in ordered_results
    ]
    print(_markdown_table(["Gamma", "Beta", "Accuracy"], rows))
    print("\nTop 3:")
    for rank, result in enumerate(top_results, start=1):
        print(
            f"{rank}. gamma={result.gamma:.1f}, beta={result.beta:.1f} "
            f"-> {_format_accuracy(result.accuracy)}"
        )


def _print_phase2_summary(results: list[SweepResult]) -> None:
    print("\n=== Phase 2: LR × θ_decay Sweep (Top 3 Phase 1 configs) ===\n")
    rows = [
        [
            f"{result.gamma:.1f}",
            f"{result.beta:.1f}",
            f"{result.lr:.3f}".rstrip("0").rstrip("."),
            f"{result.theta_decay:.3f}".rstrip("0").rstrip("."),
            _format_accuracy(result.accuracy),
        ]
        for result in results
    ]
    print(_markdown_table(["Gamma", "Beta", "LR", "θ_decay", "Accuracy"], rows))


def _run_jobs(
    jobs: list[tuple[float, float, float, float]],
    *,
    label: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    prototype_images: torch.Tensor,
    prototype_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    passes: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_workers: int,
    worker_threads: int,
) -> list[SweepResult]:
    if device.type != "cpu" or max_workers <= 1:
        results: list[SweepResult] = []
        total = len(jobs)
        for index, (gamma, beta, lr, theta_decay) in enumerate(jobs, start=1):
            started = time.perf_counter()
            result = _evaluate_config(
                gamma=gamma,
                beta=beta,
                lr=lr,
                theta_decay=theta_decay,
                train_images=train_images,
                train_labels=train_labels,
                prototype_images=prototype_images,
                prototype_labels=prototype_labels,
                test_images=test_images,
                test_labels=test_labels,
                passes=passes,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            results.append(result)
            elapsed = time.perf_counter() - started
            print(
                f"[{label} {index:02d}/{total}] gamma={gamma:.1f} beta={beta:.1f} "
                f"lr={lr:.3f} theta_decay={theta_decay:.3f} "
                f"acc={_format_accuracy(result.accuracy)} elapsed={elapsed:.1f}s"
            )
        return results

    for tensor in (train_images, train_labels, prototype_images, prototype_labels, test_images, test_labels):
        tensor.share_memory_()

    results: list[SweepResult] = []
    total = len(jobs)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(
            train_images,
            train_labels,
            prototype_images,
            prototype_labels,
            test_images,
            test_labels,
            passes,
            batch_size,
            seed,
            device.type,
            worker_threads,
        ),
    ) as executor:
        future_to_job = {
            executor.submit(_evaluate_worker, job): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(future_to_job), start=1):
            gamma, beta, lr, theta_decay = future_to_job[future]
            result, elapsed = future.result()
            results.append(result)
            print(
                f"[{label} {index:02d}/{total}] gamma={gamma:.1f} beta={beta:.1f} "
                f"lr={lr:.3f} theta_decay={theta_decay:.3f} "
                f"acc={_format_accuracy(result.accuracy)} elapsed={elapsed:.1f}s"
            )
    return results


def main() -> int:
    args = parse_args()
    _validate_args(args)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    )

    device = _resolve_device(args.device)
    print("Loading real CIFAR-10 via torchvision...")
    train_images, train_labels, test_images, test_labels = _load_real_cifar10(
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    prototype_images, prototype_labels = _balanced_tensor_subset(
        train_images,
        train_labels,
        total_samples=args.prototype_samples,
        seed=args.seed + 23,
    )
    available_threads = args.num_threads if args.num_threads > 0 else torch.get_num_threads()
    max_workers = 1 if device.type != "cpu" else min(args.max_workers, max(1, os.cpu_count() or 1))
    worker_threads = args.worker_threads
    if max_workers > 1 and worker_threads == 0:
        worker_threads = max(1, available_threads // max_workers)
    print(
        f"Device: {device.type} | train={train_images.shape[0]} | test={test_images.shape[0]} | "
        f"prototype={prototype_images.shape[0]} | passes={args.passes} | batch={args.batch_size} | "
        f"workers={max_workers} | worker_threads={worker_threads or available_threads}"
    )

    phase1_start = time.perf_counter()
    phase1_jobs = [
        (gamma, beta, DEFAULT_PHASE1_LR, DEFAULT_PHASE1_THETA_DECAY)
        for gamma in PHASE1_GAMMAS
        for beta in PHASE1_BETAS
    ]
    phase1_results = _run_jobs(
        phase1_jobs,
        label="phase1",
        train_images=train_images,
        train_labels=train_labels,
        prototype_images=prototype_images,
        prototype_labels=prototype_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=args.passes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        max_workers=max_workers,
        worker_threads=worker_threads,
    )

    ranked_phase1 = sorted(
        phase1_results,
        key=lambda result: (result.accuracy, result.gamma, result.beta),
        reverse=True,
    )
    top_results = ranked_phase1[: min(args.phase1_top_k, len(ranked_phase1))]
    _print_phase1_summary(phase1_results, top_results)
    print(f"\nPhase 1 runtime: {time.perf_counter() - phase1_start:.1f}s")

    phase2_start = time.perf_counter()
    phase2_jobs = [
        (seed_result.gamma, seed_result.beta, lr, theta_decay)
        for seed_result in top_results
        for lr in PHASE2_LRS
        for theta_decay in PHASE2_THETA_DECAYS
    ]
    phase2_results = _run_jobs(
        phase2_jobs,
        label="phase2",
        train_images=train_images,
        train_labels=train_labels,
        prototype_images=prototype_images,
        prototype_labels=prototype_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=args.passes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        max_workers=max_workers,
        worker_threads=worker_threads,
    )

    phase2_results.sort(
        key=lambda result: (result.accuracy, result.gamma, result.beta, result.lr, result.theta_decay),
        reverse=True,
    )
    _print_phase2_summary(phase2_results)
    print(f"\nPhase 2 runtime: {time.perf_counter() - phase2_start:.1f}s")

    best = phase2_results[0] if phase2_results else ranked_phase1[0]
    print(
        "\nBEST CONFIG: "
        f"gamma={best.gamma:.1f}, beta={best.beta:.1f}, lr={best.lr:.3f}, "
        f"theta_decay={best.theta_decay:.3f}, accuracy={_format_accuracy(best.accuracy)}"
    )
    print(
        "RECOMMENDED CONFIG: "
        "ConvCCCConfig("
        "softhebb_enabled=True, "
        f"softhebb_gamma={best.gamma:.1f}, "
        f"softhebb_beta={best.beta:.1f}, "
        f"softhebb_theta_decay={best.theta_decay:.3f}, "
        f"conv_hebbian_lr={best.lr:.3f}, "
        "num_conv_features=64, spatial_size=32, f1_top_k=32)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
