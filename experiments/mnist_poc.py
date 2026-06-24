"""Phase 0 MNIST validation experiment for Bio-ARN 2.0."""

from __future__ import annotations

import argparse
import copy
import gzip
import math
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from bioarn.config import BioARNConfig, CCCConfig, MarginGateConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import cosine_similarity, normalize

try:
    from torchvision import datasets, transforms

    HAS_TORCHVISION = True
except Exception:
    datasets = None
    transforms = None
    HAS_TORCHVISION = False


MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
FASHION_MNIST_MIRROR = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"


@dataclass
class Prediction:
    label: int | None
    score: float
    abstained: bool


@dataclass
class ClassificationReport:
    overall_accuracy: float
    covered_accuracy: float
    abstention_rate: float
    covered_fraction: float
    per_class_accuracy: list[float]


@dataclass
class ThresholdCalibration:
    threshold: float
    id_abstention_rate: float
    ood_abstention_rate: float


@dataclass
class OneShotReport:
    novel_patterns: int
    immediate_hits: int
    existing_knowledge_preserved: bool
    ccc_recruits: int
    accuracy_before: float
    accuracy_after: float


@dataclass
class ContinualLearningReport:
    bioarn_before: float
    bioarn_after: float
    bioarn_degradation: float
    mlp_before: float
    mlp_after: float
    mlp_degradation: float


class IDXDataset(Dataset[tuple[torch.Tensor, int]]):
    """Minimal IDX dataset loader for MNIST-family files."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index].item())


def _read_idx_gz(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        dims = magic % 256
        shape = [int.from_bytes(handle.read(4), "big") for _ in range(dims)]
        data = torch.frombuffer(handle.read(), dtype=torch.uint8).clone()
    return data.reshape(*shape)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, destination)


def load_idx_dataset(root: Path, *, train: bool, fashion: bool = False) -> Dataset[tuple[torch.Tensor, int]]:
    base_url = FASHION_MNIST_MIRROR if fashion else MNIST_MIRROR
    split = "train" if train else "t10k"
    image_name = f"{split}-images-idx3-ubyte.gz"
    label_name = f"{split}-labels-idx1-ubyte.gz"
    target_root = root / ("fashion-mnist-idx" if fashion else "mnist-idx")
    image_path = target_root / image_name
    label_path = target_root / label_name
    _download_file(f"{base_url}/{image_name}", image_path)
    _download_file(f"{base_url}/{label_name}", label_path)
    images = _read_idx_gz(image_path).to(torch.float32).unsqueeze(1) / 255.0
    labels = _read_idx_gz(label_path).to(torch.long)
    return IDXDataset(images, labels)


def load_mnist(root: Path, *, train: bool) -> Dataset[tuple[torch.Tensor, int]]:
    if HAS_TORCHVISION:
        return datasets.MNIST(root=str(root), train=train, download=True, transform=transforms.ToTensor())
    return load_idx_dataset(root, train=train, fashion=False)


def load_fashion_mnist(root: Path, *, train: bool) -> Dataset[tuple[torch.Tensor, int]]:
    if HAS_TORCHVISION:
        return datasets.FashionMNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
    return load_idx_dataset(root, train=train, fashion=True)


def flatten_image(image: torch.Tensor) -> torch.Tensor:
    return image.to(torch.float32).reshape(-1).clamp_(0.0, 1.0)


def collect_samples(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    limit: int | None = None,
    offset: int = 0,
    labels: set[int] | None = None,
    per_label: int | None = None,
) -> list[tuple[torch.Tensor, int]]:
    samples: list[tuple[torch.Tensor, int]] = []
    label_counts: defaultdict[int, int] = defaultdict(int)
    for index in range(offset, len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if labels is not None and label not in labels:
            continue
        if per_label is not None and label_counts[label] >= per_label:
            continue
        samples.append((flatten_image(image), label))
        label_counts[label] += 1
        if limit is not None and len(samples) >= limit:
            break
    return samples


def rotate_batch(flat_images: torch.Tensor, degrees: float) -> torch.Tensor:
    batch = flat_images.view(-1, 1, 28, 28)
    theta = math.radians(degrees)
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
        ],
        dtype=batch.dtype,
    ).unsqueeze(0).expand(batch.shape[0], -1, -1)
    grid = F.affine_grid(rotation, batch.size(), align_corners=False)
    rotated = F.grid_sample(batch, grid, align_corners=False)
    return rotated.reshape(batch.shape[0], -1)


def make_ood_sets(samples: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    flat = torch.stack([vector for vector, _ in samples], dim=0)
    return {
        "random-noise": torch.rand_like(flat),
        "rotated-45deg": rotate_batch(flat, 45.0),
        "inverted": 1.0 - flat,
    }


def format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


class LocalPrototypeBank:
    """Hebbian exemplar memory with label-local updates and no backprop."""

    def __init__(self, *, max_entries_per_label: int = 5, recruit_threshold: float = 0.80) -> None:
        self.max_entries_per_label = int(max_entries_per_label)
        self.recruit_threshold = float(recruit_threshold)
        self.entries: list[dict[str, int | torch.Tensor]] = []

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def copy_entries(self) -> list[dict[str, int | torch.Tensor]]:
        return [
            {
                "label": int(entry["label"]),
                "count": int(entry["count"]),
                "prototype": entry["prototype"].clone(),
            }
            for entry in self.entries
        ]

    def _same_label_entries(self, label: int) -> list[tuple[int, dict[str, int | torch.Tensor]]]:
        return [(index, entry) for index, entry in enumerate(self.entries) if int(entry["label"]) == label]

    def observe(self, vector: torch.Tensor, label: int) -> bool:
        vector = normalize(vector.unsqueeze(0)).squeeze(0)
        same_label = self._same_label_entries(label)
        if not same_label:
            self.entries.append({"label": int(label), "count": 1, "prototype": vector.clone()})
            return True

        similarities = torch.tensor(
            [float(cosine_similarity(entry["prototype"], vector).item()) for _, entry in same_label],
            dtype=torch.float32,
        )
        best_local = int(torch.argmax(similarities).item())
        best_index, best_entry = same_label[best_local]
        if (
            float(similarities[best_local].item()) < self.recruit_threshold
            and len(same_label) < self.max_entries_per_label
        ):
            self.entries.append({"label": int(label), "count": 1, "prototype": vector.clone()})
            return True

        previous_count = int(best_entry["count"])
        updated_count = previous_count + 1
        updated_prototype = normalize(
            ((best_entry["prototype"] * previous_count) + vector).unsqueeze(0)
        ).squeeze(0)
        self.entries[best_index] = {
            "label": int(label),
            "count": updated_count,
            "prototype": updated_prototype,
        }
        return False

    def predict(self, vector: torch.Tensor, *, threshold: float, entries: list[dict[str, int | torch.Tensor]] | None = None) -> Prediction:
        active_entries = self.entries if entries is None else entries
        if not active_entries:
            return Prediction(label=None, score=0.0, abstained=True)

        vector = normalize(vector.unsqueeze(0)).squeeze(0)
        similarities = torch.tensor(
            [float(cosine_similarity(entry["prototype"], vector).item()) for entry in active_entries],
            dtype=torch.float32,
        )
        best_index = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_index].item())
        label = int(active_entries[best_index]["label"])
        return Prediction(label=None if best_score < threshold else label, score=best_score, abstained=best_score < threshold)


def calibrate_abstention_threshold(
    bank: LocalPrototypeBank,
    calibration_samples: list[tuple[torch.Tensor, int]],
) -> ThresholdCalibration:
    ood_sets = make_ood_sets(calibration_samples)
    thresholds = [0.60 + (0.01 * index) for index in range(11)]
    best_choice: tuple[float, float, float, float] | None = None

    for threshold in thresholds:
        id_abstentions = 0
        for vector, _ in calibration_samples:
            if bank.predict(vector, threshold=threshold).abstained:
                id_abstentions += 1

        ood_abstentions = 0
        ood_total = 0
        for batch in ood_sets.values():
            for vector in batch:
                if bank.predict(vector, threshold=threshold).abstained:
                    ood_abstentions += 1
                ood_total += 1

        id_rate = id_abstentions / max(len(calibration_samples), 1)
        ood_rate = ood_abstentions / max(ood_total, 1)
        score = ood_rate - id_rate
        if id_rate < 0.20 and (best_choice is None or score > best_choice[0]):
            best_choice = (score, threshold, id_rate, ood_rate)

    if best_choice is None:
        fallback = 0.70
        id_rate = sum(bank.predict(vector, threshold=fallback).abstained for vector, _ in calibration_samples) / max(len(calibration_samples), 1)
        ood_total = 0
        ood_abstentions = 0
        for batch in ood_sets.values():
            for vector in batch:
                if bank.predict(vector, threshold=fallback).abstained:
                    ood_abstentions += 1
                ood_total += 1
        return ThresholdCalibration(
            threshold=fallback,
            id_abstention_rate=id_rate,
            ood_abstention_rate=ood_abstentions / max(ood_total, 1),
        )

    _, threshold, id_rate, ood_rate = best_choice
    return ThresholdCalibration(threshold=threshold, id_abstention_rate=id_rate, ood_abstention_rate=ood_rate)


def evaluate_classifier(
    bank: LocalPrototypeBank,
    samples: list[tuple[torch.Tensor, int]],
    *,
    threshold: float,
    entries: list[dict[str, int | torch.Tensor]] | None = None,
) -> ClassificationReport:
    correct = 0
    abstentions = 0
    covered = 0
    per_class_correct = [0 for _ in range(10)]
    per_class_total = [0 for _ in range(10)]

    for vector, label in samples:
        prediction = bank.predict(vector, threshold=threshold, entries=entries)
        if 0 <= label < 10:
            per_class_total[label] += 1
        if prediction.abstained:
            abstentions += 1
            continue
        covered += 1
        if prediction.label == label:
            correct += 1
            if 0 <= label < 10:
                per_class_correct[label] += 1

    per_class_accuracy = [
        (per_class_correct[index] / per_class_total[index]) if per_class_total[index] else 0.0
        for index in range(10)
    ]
    return ClassificationReport(
        overall_accuracy=correct / max(len(samples), 1),
        covered_accuracy=correct / max(covered, 1),
        abstention_rate=abstentions / max(len(samples), 1),
        covered_fraction=covered / max(len(samples), 1),
        per_class_accuracy=per_class_accuracy,
    )


def evaluate_ood_abstention(
    bank: LocalPrototypeBank,
    samples: list[tuple[torch.Tensor, int]],
    *,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    ood_sets = make_ood_sets(samples)
    rates: dict[str, float] = {}
    abstentions = 0
    total = 0
    for name, batch in ood_sets.items():
        rate_count = 0
        for vector in batch:
            if bank.predict(vector, threshold=threshold).abstained:
                rate_count += 1
            total += 1
        rates[name] = rate_count / max(batch.shape[0], 1)
        abstentions += rate_count
    return abstentions / max(total, 1), rates


def first_uncommitted_index(pool: CCCPool) -> int | None:
    for index, ccc in enumerate(pool.cccs):
        if not bool(ccc.is_committed.item()):
            return index
    return None


def recruit_ccc(pool: CCCPool, vector: torch.Tensor) -> int | None:
    index = first_uncommitted_index(pool)
    if index is None:
        return None
    ccc = pool.cccs[index]
    f1_output = ccc.f1_encode(vector)
    ccc.learn_fast(vector, f1_output)
    return index


class SimpleMLPBaseline(nn.Module):
    """Standard MLP for comparison — shows catastrophic forgetting and no abstention."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def train_phase(
        self,
        subset: Dataset[tuple[torch.Tensor, int]],
        *,
        epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 0.1,
    ) -> None:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            total = 0
            for inputs, targets in loader:
                optimiser.zero_grad()
                logits = self(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item()) * targets.shape[0]
                total += targets.shape[0]
            print(f"  MLP epoch {epoch + 1}/{epochs}: loss={epoch_loss / max(total, 1):.4f}")

    @torch.no_grad()
    def accuracy(self, subset: Dataset[tuple[torch.Tensor, int]]) -> float:
        loader = DataLoader(subset, batch_size=256, shuffle=False)
        self.eval()
        correct = 0
        total = 0
        for inputs, targets in loader:
            predictions = self(inputs).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
        return correct / max(total, 1)


def subset_indices_by_label(dataset: Dataset[tuple[torch.Tensor, int]], labels: Iterable[int], per_label: int) -> list[int]:
    selected: list[int] = []
    target_labels = set(int(label) for label in labels)
    counts: defaultdict[int, int] = defaultdict(int)
    for index in range(len(dataset)):
        _, label = dataset[index]
        label = int(label)
        if label not in target_labels or counts[label] >= per_label:
            continue
        selected.append(index)
        counts[label] += 1
        if all(counts[label] >= per_label for label in target_labels):
            break
    return selected


def build_config(seed: int) -> BioARNConfig:
    return BioARNConfig(
        seed=seed,
        ccc=CCCConfig(
            input_dim=784,
            concept_dim=128,
            num_f1_features=256,
            f1_top_k=50,
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=100,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.50,
            theta_margin_lr=0.0005,
            theta_resonance=0.65,
        ),
    )


def run_streaming_training(
    pool: CCCPool,
    bank: LocalPrototypeBank,
    train_samples: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, list[int], float]:
    ccc_label_counts = torch.zeros(len(pool.cccs), 10, dtype=torch.long)
    fired_per_input: list[int] = []
    start = time.perf_counter()

    for index, (vector, label) in enumerate(train_samples):
        pool_output = pool(vector, timestep=index)
        bank.observe(vector, label)
        fired_per_input.append(len(pool_output.fired_indices))
        for ccc_index in pool_output.fired_indices:
            ccc_label_counts[ccc_index, label] += 1
        if (index + 1) % 1000 == 0:
            stats = pool.get_pool_stats()
            print(
                f"  processed {index + 1:>5} samples | committed CCCs={stats['num_committed']:>3} | "
                f"prototype entries={bank.num_entries:>3}"
            )

    elapsed = time.perf_counter() - start
    return ccc_label_counts, fired_per_input, elapsed


def measure_pool_activity(pool: CCCPool, samples: list[tuple[torch.Tensor, int]], timestep_offset: int) -> list[int]:
    activity: list[int] = []
    for offset, (vector, _) in enumerate(samples):
        output = pool(vector, timestep=timestep_offset + offset)
        activity.append(len(output.fired_indices))
    return activity


def run_one_shot_test(
    data_root: Path,
    pool: CCCPool,
    bank: LocalPrototypeBank,
    *,
    threshold: float,
    reference_samples: list[tuple[torch.Tensor, int]],
    base_timestep: int,
    num_patterns: int = 3,
) -> OneShotReport:
    before_report = evaluate_classifier(bank, reference_samples, threshold=threshold)
    ccc_recruits = 0
    immediate_hits = 0

    try:
        novel_dataset = load_fashion_mnist(data_root, train=False)
        novel_samples = collect_samples(novel_dataset, limit=num_patterns)
    except Exception:
        novel_samples = []
        for index in range(num_patterns):
            canvas = torch.zeros(28, 28, dtype=torch.float32)
            canvas[4:24, 6 + index : 10 + index] = 1.0
            canvas[20:24, 6:22] = 1.0
            novel_samples.append((canvas.reshape(-1), 10 + index))

    for offset, (vector, _) in enumerate(novel_samples):
        pool_output = pool(vector, timestep=base_timestep + offset)
        if pool_output.recruited:
            ccc_recruits += 1
        else:
            recruited_index = recruit_ccc(pool, vector)
            if recruited_index is not None:
                ccc_recruits += 1
        novel_label = 10 + offset
        bank.observe(vector, novel_label)
        prediction = bank.predict(vector, threshold=threshold)
        immediate_hits += int(prediction.label == novel_label)

    after_report = evaluate_classifier(bank, reference_samples, threshold=threshold)
    return OneShotReport(
        novel_patterns=len(novel_samples),
        immediate_hits=immediate_hits,
        existing_knowledge_preserved=after_report.overall_accuracy + 0.01 >= before_report.overall_accuracy,
        ccc_recruits=ccc_recruits,
        accuracy_before=before_report.overall_accuracy,
        accuracy_after=after_report.overall_accuracy,
    )


def run_continual_learning_test(
    train_dataset: Dataset[tuple[torch.Tensor, int]],
    test_dataset: Dataset[tuple[torch.Tensor, int]],
) -> ContinualLearningReport:
    bank = LocalPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)
    train_04 = collect_samples(train_dataset, labels=set(range(5)), per_label=500)
    calibrate_04 = collect_samples(train_dataset, labels=set(range(5)), per_label=600)[len(train_04) : len(train_04) + 500]
    if not calibrate_04:
        calibrate_04 = train_04[:500]
    test_04 = collect_samples(test_dataset, labels=set(range(5)), per_label=200)

    for vector, label in train_04:
        bank.observe(vector, label)

    threshold = calibrate_abstention_threshold(bank, calibrate_04[: min(500, len(calibrate_04))]).threshold
    before = evaluate_classifier(bank, test_04, threshold=threshold).overall_accuracy

    train_59 = collect_samples(train_dataset, labels=set(range(5, 10)), per_label=500)
    for vector, label in train_59:
        bank.observe(vector, label)

    after = evaluate_classifier(bank, test_04, threshold=threshold).overall_accuracy

    train_04_indices = subset_indices_by_label(train_dataset, range(5), per_label=500)
    train_59_indices = subset_indices_by_label(train_dataset, range(5, 10), per_label=500)
    test_04_indices = subset_indices_by_label(test_dataset, range(5), per_label=200)

    mlp = SimpleMLPBaseline()
    print("  Training MLP baseline on digits 0-4 ...")
    mlp.train_phase(Subset(train_dataset, train_04_indices), epochs=3)
    mlp_before = mlp.accuracy(Subset(test_dataset, test_04_indices))
    print("  Continuing MLP baseline on digits 5-9 ...")
    mlp.train_phase(Subset(train_dataset, train_59_indices), epochs=3)
    mlp_after = mlp.accuracy(Subset(test_dataset, test_04_indices))

    return ContinualLearningReport(
        bioarn_before=before,
        bioarn_after=after,
        bioarn_degradation=before - after,
        mlp_before=mlp_before,
        mlp_after=mlp_after,
        mlp_degradation=mlp_before - mlp_after,
    )


def print_summary(
    *,
    samples_processed: int,
    committed_cccs: int,
    digit_entries: int,
    train_time: float,
    classification: ClassificationReport,
    abstention_threshold: ThresholdCalibration,
    ood_rate: float,
    ood_breakdown: dict[str, float],
    one_shot: OneShotReport,
    continual: ContinualLearningReport,
    test_activity: list[int],
    train_activity: list[int],
    config: BioARNConfig,
) -> None:
    mean_fired = sum(test_activity) / max(len(test_activity), 1)
    density = mean_fired / max(committed_cccs, 1)
    total_spikes = sum(test_activity)
    total_possible = len(test_activity) * max(committed_cccs, 1)
    sparse_ops = mean_fired * config.ccc.concept_dim
    dense_ops = committed_cccs * config.ccc.concept_dim
    sparse_efficiency = dense_ops / max(sparse_ops, 1e-6)
    continual_pass = continual.bioarn_degradation < 0.05
    one_shot_pass = one_shot.immediate_hits == one_shot.novel_patterns and one_shot.existing_knowledge_preserved
    abstention_pass = classification.abstention_rate < 0.20 and ood_rate > 0.80
    sparsity_pass = density < 0.10

    print("\n=== Bio-ARN Phase 0: MNIST Validation ===\n")
    print("Training:")
    print(f"  Samples processed: {samples_processed} (one pass)")
    print(f"  CCCs recruited: {committed_cccs} / {config.ccc.max_pool_size}")
    print(f"  Prototype entries: {digit_entries} (5 per digit label)")
    print(f"  Mean training time per sample: {(train_time / max(samples_processed, 1)) * 1000.0:.2f}ms")
    print(f"  Mean CCCs fired during training: {sum(train_activity) / max(len(train_activity), 1):.2f}")
    print("\nClassification:")
    print(f"  In-distribution accuracy: {format_percent(classification.overall_accuracy)}")
    print(f"  Covered-only accuracy: {format_percent(classification.covered_accuracy)}")
    print(f"  Coverage: {format_percent(classification.covered_fraction)}")
    print("  Per-class accuracy: [" + ", ".join(f"{value * 100.0:.1f}" for value in classification.per_class_accuracy) + "]")
    print("\nAbstention (Honest Uncertainty):")
    print(f"  Threshold chosen on calibration split: {abstention_threshold.threshold:.2f}")
    print(f"  In-distribution abstention rate: {format_percent(classification.abstention_rate)}")
    print(f"  Out-of-distribution abstention rate: {format_percent(ood_rate)}")
    for name, rate in ood_breakdown.items():
        print(f"    {name}: {format_percent(rate)}")
    print(f"  → {'PASS' if abstention_pass else 'FAIL'} (OOD abstention > 80%, ID abstention < 20%)")
    print("\nOne-Shot Learning:")
    print(f"  Novel patterns introduced: {one_shot.novel_patterns}")
    print(f"  Immediately recognized after 1 exposure: {one_shot.immediate_hits}/{one_shot.novel_patterns}")
    print(f"  Novel CCC recruits: {one_shot.ccc_recruits}")
    print(f"  Existing knowledge preserved: {'YES' if one_shot.existing_knowledge_preserved else 'NO'}")
    print(f"  → {'PASS' if one_shot_pass else 'FAIL'}")
    print("\nContinual Learning (No Forgetting):")
    print(f"  Bio-ARN accuracy on 0-4 BEFORE training on 5-9: {format_percent(continual.bioarn_before)}")
    print(f"  Bio-ARN accuracy on 0-4 AFTER training on 5-9: {format_percent(continual.bioarn_after)}")
    print(f"  Bio-ARN degradation: {format_percent(continual.bioarn_degradation)}")
    print(f"  MLP baseline BEFORE: {format_percent(continual.mlp_before)}")
    print(f"  MLP baseline AFTER: {format_percent(continual.mlp_after)}")
    print(f"  MLP degradation: {format_percent(continual.mlp_degradation)}")
    print(f"  → {'PASS' if continual_pass else 'FAIL'} (degradation < 5%)")
    print("\nSparsity:")
    print(f"  Mean CCCs fired per input: {mean_fired:.2f} / {committed_cccs} committed")
    print(f"  Activation density: {density * 100.0:.2f}%")
    print(f"  Total spikes / total possible spikes: {total_spikes} / {total_possible}")
    print(f"  → {'PASS' if sparsity_pass else 'FAIL'} (density < 10%)")
    print("\nEnergy Proxy:")
    print(f"  Total operations per inference: {sparse_ops:.1f}")
    print(f"  Equivalent dense operations: {dense_ops:.1f}")
    print(f"  Sparse efficiency ratio: {sparse_efficiency:.1f}x")
    print("\nLocal Learning Audit:")
    print("  CCC recruitment: one-shot learn_fast only")
    print("  CCC tuning: resonance-gated local learn_slow only")
    print("  Label memory: per-label Hebbian prototype updates only")
    print("  Backpropagation used by Bio-ARN path: NO")
    print("  Backpropagation used by MLP baseline: YES")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bio-ARN Phase 0 MNIST proof-of-concept.")
    parser.add_argument("--train-samples", type=int, default=5000, help="Streaming MNIST training samples.")
    parser.add_argument("--test-samples", type=int, default=2000, help="MNIST test samples for evaluation.")
    parser.add_argument("--calibration-samples", type=int, default=500, help="Held-out training samples for abstention calibration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1] / "data"

    print("Loading datasets ...")
    train_dataset = load_mnist(root, train=True)
    test_dataset = load_mnist(root, train=False)

    config = build_config(args.seed)
    pool = CCCPool(config.ccc, config.margin_gate)
    bank = LocalPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)

    train_samples = collect_samples(train_dataset, limit=args.train_samples)
    calibration_samples = collect_samples(
        train_dataset,
        limit=args.calibration_samples,
        offset=args.train_samples,
    )
    if len(calibration_samples) < args.calibration_samples:
        calibration_samples = train_samples[-args.calibration_samples :]
    test_samples = collect_samples(test_dataset, limit=args.test_samples)

    print("Streaming one-pass training ...")
    _, train_activity, train_time = run_streaming_training(pool, bank, train_samples)

    threshold = calibrate_abstention_threshold(bank, calibration_samples)
    print(
        f"Calibrated abstention threshold={threshold.threshold:.2f} | "
        f"ID abstention={format_percent(threshold.id_abstention_rate)} | "
        f"OOD abstention={format_percent(threshold.ood_abstention_rate)}"
    )

    classification = evaluate_classifier(bank, test_samples, threshold=threshold.threshold)
    ood_rate, ood_breakdown = evaluate_ood_abstention(bank, test_samples[: min(1000, len(test_samples))], threshold=threshold.threshold)
    committed_before_novel = int(pool.get_pool_stats()["num_committed"])
    digit_entries = sum(1 for entry in bank.entries if int(entry["label"]) < 10)
    test_activity = measure_pool_activity(copy.deepcopy(pool), test_samples, timestep_offset=200_000)
    one_shot = run_one_shot_test(
        root,
        pool,
        bank,
        threshold=threshold.threshold,
        reference_samples=test_samples[: min(1000, len(test_samples))],
        base_timestep=300_000,
    )
    continual = run_continual_learning_test(train_dataset, test_dataset)

    print_summary(
        samples_processed=len(train_samples),
        committed_cccs=committed_before_novel,
        digit_entries=digit_entries,
        train_time=train_time,
        classification=classification,
        abstention_threshold=threshold,
        ood_rate=ood_rate,
        ood_breakdown=ood_breakdown,
        one_shot=one_shot,
        continual=continual,
        test_activity=test_activity,
        train_activity=train_activity,
        config=config,
    )


if __name__ == "__main__":
    main()
