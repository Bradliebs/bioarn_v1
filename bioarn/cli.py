"""Bio-ARN 2.0 CLI — Train, evaluate, and serve Bio-ARN models."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from bioarn.hardware.energy_model import EnergyModel
from bioarn.hardware.deployment import LoihiDeploymentPipeline
from bioarn.loop import SensorimotorLoop
from bioarn.persistence import ModelStore
from bioarn.system import BioARNCore
from bioarn.training import OnlineTrainer
from bioarn.utils import BioARNLogger, CheckpointManager, ConfigManager, ReproducibilityManager

try:
    from torchvision import datasets, transforms

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - optional dependency
    datasets = None
    transforms = None
    _HAS_TORCHVISION = False

_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
_FASHION_MNIST_MIRROR = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bio-ARN 2.0 production CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a Bio-ARN model online")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--preset", type=str, default=None)
    train_parser.add_argument("--data", type=str, default="mnist")
    train_parser.add_argument("--output", type=str, default="models")
    train_parser.add_argument("--max-steps", type=int, default=128)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument("--checkpoint-interval", type=int, default=50)
    train_parser.add_argument("--keep-last", type=int, default=5)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.set_defaults(func=_command_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--data", type=str, default="mnist_test")
    eval_parser.add_argument("--num-samples", type=int, default=128)
    eval_parser.set_defaults(func=_command_evaluate)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a concept seed")
    generate_parser.add_argument("--checkpoint", type=str, required=True)
    generate_parser.add_argument("--prompt", type=str, required=True)
    generate_parser.add_argument("--max-tokens", type=int, default=100)
    generate_parser.set_defaults(func=_command_generate)

    profile_parser = subparsers.add_parser("profile", help="Profile sparsity, latency, and energy")
    profile_parser.add_argument("--config", type=str, default=None)
    profile_parser.add_argument("--preset", type=str, default=None)
    profile_parser.add_argument("--data", type=str, default="mnist")
    profile_parser.add_argument("--num-samples", type=int, default=100)
    profile_parser.set_defaults(func=_command_profile)

    info_parser = subparsers.add_parser("info", help="Inspect checkpoint metadata")
    info_parser.add_argument("--checkpoint", type=str, required=True)
    info_parser.set_defaults(func=_command_info)

    deploy_parser = subparsers.add_parser("deploy", help="Prepare a model for Lava/Loihi deployment")
    deploy_parser.add_argument("--model", type=str, required=True)
    deploy_parser.add_argument("--version", type=str, default="latest")
    deploy_parser.add_argument("--store", type=str, default="models")
    deploy_parser.add_argument("--output", type=str, default="deployments")
    deploy_parser.set_defaults(func=_command_deploy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


def _command_train(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    ReproducibilityManager.set_seed(config.seed)
    output_dir = Path(args.output)
    logger = BioARNLogger(component="cli.train", log_dir=output_dir / "logs")
    system = SensorimotorLoop(config)
    trainer = OnlineTrainer(
        logger=logger,
        checkpoint_manager=CheckpointManager(keep_last=args.keep_last),
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_interval,
        output_dir=output_dir,
        keep_last=args.keep_last,
    )
    data = _load_named_data(args.data, limit=args.max_steps, config=config)
    result = trainer.train(system, data, config)
    logger.flush()
    print(f"Training complete: steps={result.total_steps} accuracy={result.accuracy:.3f} checkpoints={len(result.checkpoints)}")
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    manager = CheckpointManager()
    checkpoint = manager.read_checkpoint(args.checkpoint, resolve=True)
    system = manager.load(args.checkpoint)
    trainer = OnlineTrainer(output_dir=None)
    data = _load_named_data(args.data, limit=args.num_samples, config=system.config)
    result = trainer.evaluate(system, data, label_prototypes=checkpoint.get("metadata", {}).get("label_prototypes"))
    rows = {
        "accuracy": result.accuracy,
        "abstention_rate": result.abstention_rate,
        "sparsity": result.sparsity,
        "latency_ms": result.latency_ms,
        "mean_free_energy": result.mean_free_energy,
        "total_samples": result.total_samples,
    }
    _print_table(rows)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    manager = CheckpointManager()
    loaded = manager.load(args.checkpoint)
    system = _ensure_loop(loaded)
    concept_seed = _prompt_to_concept(args.prompt, system.concept_dim)
    if args.prompt:
        prompt_tokens = _tokenize_prompt(args.prompt, system.vocab_size)
        if prompt_tokens.numel() > 0:
            system.step(language_input=prompt_tokens)
    generated = system.generate_text(concept_seed, max_tokens=args.max_tokens)
    print(generated)
    return 0


def _command_profile(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    ReproducibilityManager.set_seed(config.seed)
    system = SensorimotorLoop(config)
    trainer = OnlineTrainer(output_dir=None)
    data = _load_named_data(args.data, limit=args.num_samples, config=config)
    eval_result = trainer.evaluate(system, data)
    energy = EnergyModel().estimate_inference_energy(config, "cpu_laptop", num_cccs_active=1)
    rows = {
        "samples": eval_result.total_samples,
        "accuracy": eval_result.accuracy,
        "abstention_rate": eval_result.abstention_rate,
        "sparsity": eval_result.sparsity,
        "latency_ms": eval_result.latency_ms,
        "energy_joules": energy.total_joules,
    }
    _print_table(rows)
    return 0


def _command_info(args: argparse.Namespace) -> int:
    manager = CheckpointManager()
    payload = manager.read_checkpoint(args.checkpoint, resolve=True)
    system = manager.load(args.checkpoint)
    stats = system.core.get_system_stats() if isinstance(system, SensorimotorLoop) else system.get_system_stats()
    rows = {
        "system_type": payload.get("system_type", "unknown"),
        "timestamp": payload.get("timestamp", ""),
        "training_step": payload.get("metadata", {}).get("training_step", 0),
        "concepts_learned": stats["concepts_learned"],
        "sparsity": stats["sparsity"],
        "config": json.dumps(payload.get("config", {}), sort_keys=True),
    }
    metrics = payload.get("metadata", {}).get("metrics")
    if metrics:
        rows["metrics"] = json.dumps(metrics, sort_keys=True)
    _print_table(rows)
    return 0


def _command_deploy(args: argparse.Namespace) -> int:
    store = ModelStore(args.store)
    pipeline = LoihiDeploymentPipeline(store)
    package = pipeline.prepare_for_deployment(args.model, args.version)

    output_dir = Path(args.output) / package.model_name / f"v{package.version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    export = store.export_for_loihi(package.model_name, package.version, str(output_dir / "loihi-export"))
    config_path = output_dir / "deployment_config.json"
    summary_path = output_dir / "deployment_summary.json"
    config_path.write_text(json.dumps(package.config, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "model_name": package.model_name,
                "version": package.version,
                "ready_for_hardware": package.ready_for_hardware,
                "quantization_bits": package.quantization_bits,
                "match_rate": package.equivalence.match_rate,
                "max_deviation": package.equivalence.max_deviation,
                "num_cores": package.hardware_reqs.num_cores,
                "estimated_power_mw": package.hardware_reqs.estimated_power_mw,
                "estimated_latency_ms": package.hardware_reqs.estimated_latency_ms,
                "export_dir": export.output_dir,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"Deployment package ready={package.ready_for_hardware} "
        f"cores={package.hardware_reqs.num_cores} config={config_path}"
    )
    return 0


def _resolve_config(args: argparse.Namespace):
    return ConfigManager.load(
        file_path=args.config,
        preset=args.preset,
        cli_args={"device": getattr(args, "device", None), "seed": getattr(args, "seed", None)},
    )


def _ensure_loop(system: BioARNCore | SensorimotorLoop) -> SensorimotorLoop:
    if isinstance(system, SensorimotorLoop):
        return system
    loop = SensorimotorLoop(system.config)
    loop.core = system
    loop.connector.ccc_pool = loop.core.ccc_pool
    return loop


def _tokenize_prompt(prompt: str, vocab_size: int) -> torch.Tensor:
    if not prompt:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([ord(char) % vocab_size for char in prompt], dtype=torch.long)


def _prompt_to_concept(prompt: str, concept_dim: int) -> torch.Tensor:
    if not prompt:
        return torch.zeros(concept_dim, dtype=torch.float32)
    concept = torch.zeros(concept_dim, dtype=torch.float32)
    for index, char in enumerate(prompt.encode("utf-8")):
        concept[index % concept_dim] += float((char % 17) + 1)
    if torch.count_nonzero(concept).item() == 0:
        return concept
    return torch.nn.functional.normalize(concept.unsqueeze(0), dim=-1).squeeze(0)


def _print_table(rows: dict[str, Any]) -> None:
    for key, value in rows.items():
        if isinstance(value, float):
            display = f"{value:.6f}"
        else:
            display = str(value)
        print(f"{key:>20}: {display}")


def _load_named_data(name: str, *, limit: int, config: Any) -> list[tuple[torch.Tensor, int]]:
    normalized = name.lower()
    if normalized in {"mnist", "mnist_test", "fashion_mnist", "fashion_mnist_test"}:
        train = not normalized.endswith("_test")
        fashion = normalized.startswith("fashion")
        return _load_mnist_family(train=train, fashion=fashion, limit=limit)
    if normalized in {"cifar", "cifar_test"}:
        train = normalized == "cifar"
        return _load_cifar(train=train, limit=limit, input_dim=config.ccc.input_dim)
    if normalized.startswith("language"):
        return _load_language(limit=limit)
    return _load_synthetic(limit=limit, input_dim=config.ccc.input_dim)


def _load_mnist_family(*, train: bool, fashion: bool, limit: int) -> list[tuple[torch.Tensor, int]]:
    if _HAS_TORCHVISION:
        dataset_cls = datasets.FashionMNIST if fashion else datasets.MNIST
        dataset = dataset_cls(
            root=str(Path("data")),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        return [
            (image.reshape(-1).to(torch.float32), int(label))
            for image, label in (dataset[index] for index in range(min(limit, len(dataset))))
        ]
    return _load_idx_dataset(train=train, fashion=fashion, limit=limit)


def _load_idx_dataset(*, train: bool, fashion: bool, limit: int) -> list[tuple[torch.Tensor, int]]:
    base_url = _FASHION_MNIST_MIRROR if fashion else _MNIST_MIRROR
    split = "train" if train else "t10k"
    root = Path("data") / ("fashion-mnist-idx" if fashion else "mnist-idx")
    image_path = root / f"{split}-images-idx3-ubyte.gz"
    label_path = root / f"{split}-labels-idx1-ubyte.gz"
    _download_if_needed(f"{base_url}/{image_path.name}", image_path)
    _download_if_needed(f"{base_url}/{label_path.name}", label_path)
    images = _read_idx_gz(image_path).to(torch.float32).reshape(-1, 28 * 28) / 255.0
    labels = _read_idx_gz(label_path).to(torch.long)
    return [
        (images[index], int(labels[index].item()))
        for index in range(min(limit, labels.numel()))
    ]


def _load_cifar(*, train: bool, limit: int, input_dim: int) -> list[tuple[torch.Tensor, int]]:
    if _HAS_TORCHVISION:
        dataset = datasets.CIFAR10(
            root=str(Path("data")),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        return [
            (image.reshape(-1).to(torch.float32), int(label))
            for image, label in (dataset[index] for index in range(min(limit, len(dataset))))
        ]
    generator = torch.Generator().manual_seed(123 if train else 456)
    return [
        (torch.rand(input_dim, generator=generator), index % 10)
        for index in range(limit)
    ]


def _load_language(*, limit: int) -> list[tuple[torch.Tensor, int]]:
    corpus = [
        "hello world",
        "bio arn",
        "predictive coding",
        "global workspace",
        "sparse memory",
    ]
    samples: list[tuple[torch.Tensor, int]] = []
    for index in range(limit):
        text = corpus[index % len(corpus)]
        tokens = torch.tensor([ord(char) % 64 for char in text], dtype=torch.long)
        samples.append((tokens, index % len(corpus)))
    return samples


def _load_synthetic(*, limit: int, input_dim: int) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(42)
    return [
        (torch.rand(input_dim, generator=generator), index % 10)
        for index in range(limit)
    ]


def _read_idx_gz(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        dims = magic % 256
        shape = [int.from_bytes(handle.read(4), "big") for _ in range(dims)]
        data = torch.frombuffer(handle.read(), dtype=torch.uint8).clone()
    return data.reshape(*shape)


def _download_if_needed(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    import urllib.request

    urllib.request.urlretrieve(url, destination)


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


__all__ = ["build_parser", "main"]
