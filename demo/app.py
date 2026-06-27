"""Production Gradio demo for Bio-ARN 2.0."""

from __future__ import annotations

import io
import os
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Gradio is required for the demo. Install dependencies with `pip install -r demo/requirements.txt`."
    ) from exc

from bioarn.config import PrecisionConfig  # noqa: E402
from bioarn.data import MNISTStream  # noqa: E402
from bioarn.training import VisionTrainConfig, VisionTrainer  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

MODEL_CACHE_DIR = Path(__file__).with_name("model_cache")
MODEL_CACHE_PATH = MODEL_CACHE_DIR / "mnist_demo_v1.pt"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "figure1_full_bioarn_pipeline.png"
README_PAPER_URL = "https://github.com/bioarn/bioarn/blob/main/BioARN_Architecture.md"
README_GITHUB_URL = "https://github.com/bioarn/bioarn"

TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 60
CONTINUAL_TRAIN_PER_CLASS = 40
CONTINUAL_TEST_PER_CLASS = 20
OOD_THRESHOLD = 0.55

Sample = tuple[torch.Tensor, int]


@dataclass
class DemoBackend:
    trainer: VisionTrainer
    source: str
    train_samples: list[Sample]
    test_samples: list[Sample]
    class_centroids: dict[int, torch.Tensor]
    class_counts: dict[int, int]
    label_examples: dict[int, torch.Tensor]
    train_summary: dict[str, object]
    eval_summary: dict[str, object]


@dataclass
class OnlineLearningState:
    trainer: VisionTrainer
    label_names: dict[int, str]
    label_to_id: dict[str, int]
    class_centroids: dict[int, torch.Tensor]
    class_counts: dict[int, int]
    label_examples: dict[int, torch.Tensor]
    events: list[dict[str, Any]]


@dataclass
class ContinualDemoState:
    trainers: dict[str, VisionTrainer]
    stage_index: int
    task_train: list[list[Sample]]
    task_test: list[list[Sample]]
    history: dict[str, list[list[float]]]
    source: str


def _clone_with_torch(obj: Any) -> Any:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _normalize_vector(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.to(torch.float32).reshape(-1)
    if float(flat.norm().item()) == 0.0:
        return flat
    return F.normalize(flat.unsqueeze(0), dim=-1).squeeze(0)


def _pick_image(drawn_image: Any, uploaded_image: Any) -> Any:
    if isinstance(drawn_image, dict):
        if drawn_image.get("composite") is not None or drawn_image.get("background") is not None:
            return drawn_image
        if any(layer is not None for layer in drawn_image.get("layers", [])):
            return drawn_image
    elif drawn_image is not None:
        return drawn_image
    return uploaded_image


def _coerce_image_tensor(image: Any, *, size: int = 28) -> torch.Tensor:
    if image is None:
        raise ValueError("Please upload or draw an image.")

    if isinstance(image, dict):
        layered_image = (
            image.get("composite")
            or image.get("background")
            or next((layer for layer in image.get("layers", []) if layer is not None), None)
        )
        if layered_image is None:
            raise ValueError("The canvas is empty.")
        image = layered_image

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().to(torch.float32)
    else:
        tensor = torch.tensor(np.asarray(image), dtype=torch.float32)

    if tensor.numel() == size * size:
        tensor = tensor.reshape(size, size)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] in {1, 3, 4}:
            tensor = tensor[:3].mean(dim=0)
        elif tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor[..., :3].mean(dim=-1)
        else:
            tensor = tensor.mean(dim=0)
    if tensor.dim() != 2:
        raise ValueError("Expected a 2D image.")

    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0.0, 1.0)
    if float(tensor.mean().item()) > 0.5:
        tensor = 1.0 - tensor
    if tuple(tensor.shape) != (size, size):
        tensor = F.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return tensor.to(torch.float32)


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = _coerce_image_tensor(tensor)
    return (image.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def _draw_digit_prototype(digit: int) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    segments = {
        "a": ((4, 6), (4, 22)),
        "b": ((4, 22), (14, 22)),
        "c": ((14, 22), (24, 22)),
        "d": ((24, 6), (24, 22)),
        "e": ((14, 6), (24, 6)),
        "f": ((4, 6), (14, 6)),
        "g": ((14, 6), (14, 22)),
    }
    digit_segments = {
        0: ("a", "b", "c", "d", "e", "f"),
        1: ("b", "c"),
        2: ("a", "b", "g", "e", "d"),
        3: ("a", "b", "g", "c", "d"),
        4: ("f", "g", "b", "c"),
        5: ("a", "f", "g", "c", "d"),
        6: ("a", "f", "g", "e", "c", "d"),
        7: ("a", "b", "c"),
        8: ("a", "b", "c", "d", "e", "f", "g"),
        9: ("a", "b", "c", "d", "f", "g"),
    }
    for name in digit_segments[int(digit)]:
        (r0, c0), (r1, c1) = segments[name]
        if r0 == r1:
            image[max(r0 - 1, 0) : min(r0 + 2, 28), min(c0, c1) : max(c0, c1)] = 1.0
        else:
            image[min(r0, r1) : max(r0, r1), max(c0 - 1, 0) : min(c0 + 2, 28)] = 1.0
    return image.clamp_(0.0, 1.0)


def _augment_digit(pattern: torch.Tensor, *, seed: int, noise_scale: float = 0.08) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
    shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
    rolled = torch.roll(pattern, shifts=(shift_y, shift_x), dims=(0, 1))
    noise = torch.randn(rolled.shape, generator=generator) * noise_scale
    gain = 0.85 + (0.3 * torch.rand(1, generator=generator).item())
    return (rolled * gain + noise).clamp_(0.0, 1.0)


def _synthetic_digit_dataset(
    *,
    train_per_class: int,
    test_per_class: int,
) -> tuple[list[Sample], list[Sample], dict[int, torch.Tensor]]:
    train: list[Sample] = []
    test: list[Sample] = []
    examples: dict[int, torch.Tensor] = {}
    for digit in range(10):
        prototype = _draw_digit_prototype(digit)
        examples[digit] = prototype.clone()
        for index in range(train_per_class):
            train.append((_augment_digit(prototype, seed=(digit * 10_000) + index).reshape(-1), digit))
        for index in range(test_per_class):
            test.append((_augment_digit(prototype, seed=100_000 + (digit * 10_000) + index).reshape(-1), digit))
    return train, test, examples


def _balanced_subset(stream: MNISTStream, *, per_class: int) -> tuple[list[Sample], dict[int, torch.Tensor]]:
    counts: Counter[int] = Counter()
    samples: list[Sample] = []
    examples: dict[int, torch.Tensor] = {}
    for sample in stream.stream():
        label = int(sample.label) if sample.label is not None else None
        if label is None or counts[label] >= per_class:
            continue
        tensor = sample.data.to(torch.float32).reshape(-1).cpu().clone()
        samples.append((tensor, label))
        examples.setdefault(label, tensor.reshape(28, 28).clone())
        counts[label] += 1
        if len(examples) == 10 and all(counts[digit] >= per_class for digit in range(10)):
            return samples, examples
    raise RuntimeError(f"Unable to collect a balanced {per_class}-per-class MNIST subset.")


def _load_demo_datasets() -> tuple[str, list[Sample], list[Sample], dict[int, torch.Tensor]]:
    try:
        train_stream = MNISTStream(
            split="train",
            data_dir=PROJECT_ROOT / "data",
            flatten=True,
            normalize=True,
            shuffle=True,
            seed=7,
        )
        test_stream = MNISTStream(
            split="test",
            data_dir=PROJECT_ROOT / "data",
            flatten=True,
            normalize=True,
            shuffle=True,
            seed=11,
        )
        train_samples, train_examples = _balanced_subset(train_stream, per_class=TRAIN_PER_CLASS)
        test_samples, test_examples = _balanced_subset(test_stream, per_class=TEST_PER_CLASS)
        example_bank = {digit: test_examples.get(digit, train_examples[digit]) for digit in range(10)}
        return "MNIST subset (2,000 train / 600 test)", train_samples, test_samples, example_bank
    except Exception as exc:
        train_samples, test_samples, examples = _synthetic_digit_dataset(
            train_per_class=TRAIN_PER_CLASS,
            test_per_class=TEST_PER_CLASS,
        )
        return (
            f"Synthetic 28×28 fallback ({type(exc).__name__})",
            train_samples,
            test_samples,
            examples,
        )


def _compute_class_centroids(samples: list[Sample]) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    sums: dict[int, torch.Tensor] = {}
    counts: Counter[int] = Counter()
    for tensor, label in samples:
        label = int(label)
        flat = tensor.to(torch.float32).reshape(-1)
        sums[label] = flat.clone() if label not in sums else sums[label] + flat
        counts[label] += 1
    centroids = {
        label: _normalize_vector(total / max(counts[label], 1))
        for label, total in sums.items()
    }
    return centroids, {label: int(count) for label, count in counts.items()}


def _default_train_config(train_samples: int, test_samples: int) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=28 * 28,
        concept_dim=64,
        max_pool_size=96,
        margin_threshold=0.45,
        use_batched=False,
        batch_size=32,
        learning_rate=0.03,
        num_train_samples=train_samples,
        num_test_samples=test_samples,
        preprocessing_warmup_samples=0,
        freeze_f1_after=max(50, train_samples // 12),
        consolidation_strength=0.1,
        precision=PrecisionConfig(
            enabled=True,
            pool_size=96,
            entropy_window=32,
            precision_alpha=5.0,
            precision_threshold=0.5,
            min_precision=0.1,
            max_precision=1.0,
        ),
    )


def _build_backend() -> DemoBackend:
    source, train_samples, test_samples, examples = _load_demo_datasets()
    trainer = VisionTrainer(_default_train_config(len(train_samples), len(test_samples)))
    trainer.system.ccc_pool.config.lock_threshold = 0.8
    train_summary = trainer.train_online(
        train_samples,
        num_samples=len(train_samples),
        interleave_classes=True,
    )
    eval_summary = trainer.evaluate(test_samples, num_samples=len(test_samples))
    centroids, counts = _compute_class_centroids(train_samples)
    return DemoBackend(
        trainer=trainer,
        source=source,
        train_samples=train_samples,
        test_samples=test_samples,
        class_centroids=centroids,
        class_counts=counts,
        label_examples=examples,
        train_summary=train_summary,
        eval_summary=eval_summary,
    )


def _backend_to_payload(backend: DemoBackend) -> dict[str, Any]:
    return {
        "version": 1,
        "source": backend.source,
        "trainer": backend.trainer,
        "train_samples": backend.train_samples,
        "test_samples": backend.test_samples,
        "class_centroids": backend.class_centroids,
        "class_counts": backend.class_counts,
        "label_examples": backend.label_examples,
        "train_summary": backend.train_summary,
        "eval_summary": backend.eval_summary,
    }


def _payload_to_backend(payload: dict[str, Any]) -> DemoBackend:
    return DemoBackend(
        trainer=payload["trainer"],
        source=str(payload["source"]),
        train_samples=list(payload["train_samples"]),
        test_samples=list(payload["test_samples"]),
        class_centroids=dict(payload["class_centroids"]),
        class_counts={int(key): int(value) for key, value in dict(payload["class_counts"]).items()},
        label_examples=dict(payload["label_examples"]),
        train_summary=dict(payload["train_summary"]),
        eval_summary=dict(payload["eval_summary"]),
    )


@lru_cache(maxsize=1)
def get_demo_backend() -> DemoBackend:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_CACHE_PATH.exists():
        try:
            payload = torch.load(MODEL_CACHE_PATH, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and payload.get("version") == 1:
                return _payload_to_backend(payload)
        except Exception:
            pass
    backend = _build_backend()
    torch.save(_backend_to_payload(backend), MODEL_CACHE_PATH)
    return backend


def _count_locked_cccs(pool: object) -> int:
    pool_stats = getattr(pool, "get_pool_stats", lambda: {})()
    if "num_locked" in pool_stats:
        return int(pool_stats["num_locked"])
    cccs = getattr(pool, "cccs", None)
    if cccs is None:
        return 0
    count = 0
    for ccc in cccs:
        if hasattr(ccc, "locked"):
            count += int(bool(ccc.locked.item()))
        elif hasattr(ccc, "is_locked"):
            count += int(bool(ccc.is_locked.item()))
    return count


def _dominant_label_for_ccc(trainer: VisionTrainer, label_names: dict[int, str], ccc_index: int) -> str:
    counts = trainer.ccc_label_counts.get(int(ccc_index))
    if not counts:
        return "unassigned"
    label_id, count = counts.most_common(1)[0]
    return f"{label_names.get(int(label_id), str(label_id))} ({count} hits)"


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = _normalize_vector(left)
    right_norm = _normalize_vector(right)
    if float(left_norm.norm().item()) == 0.0 or float(right_norm.norm().item()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(left_norm.unsqueeze(0), right_norm.unsqueeze(0), dim=-1).item())


def _softmax_scores(scores: list[float], *, temperature: float) -> torch.Tensor:
    values = torch.tensor(scores, dtype=torch.float32)
    if values.numel() == 0:
        return values
    return torch.softmax(values * float(temperature), dim=0)


def _ccc_visual_html(
    fired_rows: list[dict[str, Any]],
    *,
    active_cccs: int,
    locked_cccs: int,
    committed_cccs: int,
) -> str:
    if not fired_rows:
        bars = "<div style='padding:10px;border:1px dashed #cbd5e1;border-radius:10px;color:#64748b'>No CCC fired strongly enough.</div>"
    else:
        bar_rows = []
        for row in fired_rows:
            confidence_pct = max(1.0, min(100.0, float(row["confidence"]) * 100.0))
            bar_rows.append(
                "<div style='margin:8px 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.92rem'><span>CCC {row['ccc_id']} · {row['label']}</span><span>{float(row['confidence']):.3f}</span></div>"
                f"<div style='height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden'><div style='width:{confidence_pct:.1f}%;height:100%;background:linear-gradient(90deg,#2563eb,#06b6d4)'></div></div>"
                "</div>"
            )
        bars = "".join(bar_rows)
    return (
        "<div style='border:1px solid #dbeafe;border-radius:14px;padding:14px;background:#f8fbff'>"
        "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;font-size:0.95rem'>"
        f"<span><b>Active:</b> {active_cccs}</span>"
        f"<span><b>Locked:</b> {locked_cccs}</span>"
        f"<span><b>Committed:</b> {committed_cccs}</span>"
        "</div>"
        f"{bars}"
        "</div>"
    )


def _classify_tensor(
    trainer: VisionTrainer,
    tensor: torch.Tensor,
    *,
    class_centroids: dict[int, torch.Tensor],
    label_examples: dict[int, torch.Tensor],
    label_names: dict[int, str],
) -> dict[str, Any]:
    flat = tensor.reshape(-1)
    prepared = trainer._prepare_tensor(flat)  # noqa: SLF001
    step_result = trainer._step_pool(prepared, allow_recruit=False, preview=True)  # noqa: SLF001

    label_ids = sorted(label_names)
    concept_scores: list[float] = []
    centroid_scores: list[float] = []
    example_scores: list[float] = []
    for label_id in label_ids:
        prototype = trainer.label_bank.prototypes.get(label_id)
        concept_scores.append(
            0.0
            if prototype is None
            else _cosine_similarity(step_result.concept_direction, prototype.reshape(-1))
        )
        centroid_scores.append(
            0.0
            if label_id not in class_centroids
            else _cosine_similarity(flat, class_centroids[label_id])
        )
        example_scores.append(
            0.0
            if label_id not in label_examples
            else _cosine_similarity(flat, label_examples[label_id].reshape(-1))
        )

    concept_probs = _softmax_scores(concept_scores, temperature=4.0)
    centroid_probs = _softmax_scores(centroid_scores, temperature=7.0)
    example_probs = _softmax_scores(example_scores, temperature=9.0)
    combined = (0.35 * concept_probs) + (0.4 * centroid_probs) + (0.25 * example_probs)
    combined = combined / combined.sum().clamp_min(1e-6)

    label_distribution = {
        label_names[label_id]: float(combined[index].item())
        for index, label_id in enumerate(label_ids)
    }
    top_values, top_indices = torch.sort(combined, descending=True)
    best_index = int(top_indices[0].item()) if top_indices.numel() > 0 else 0
    prediction_label_id = label_ids[best_index] if label_ids else -1
    prediction_name = label_names.get(prediction_label_id, str(prediction_label_id))
    top_confidence = float(top_values[0].item()) if top_values.numel() else 0.0
    margin = float(top_values[0] - top_values[1]) if top_values.numel() > 1 else top_confidence
    abstained = bool(step_result.abstained or top_confidence < 0.18 or margin < 0.03)

    prototype_alignment = max(example_scores or [0.0])
    probabilities = combined.clamp_min(1e-6)
    entropy = float(
        (-(probabilities * probabilities.log()).sum() / torch.log(torch.tensor(float(len(probabilities))))).item()
    )
    ood_score = min(
        1.0,
        max(
            0.0,
            (0.45 * (1.0 - top_confidence))
            + (0.25 * (1.0 - prototype_alignment))
            + (0.15 * entropy)
            + (0.10 if abstained else 0.0)
            + (0.05 if len(step_result.fired_indices) == 0 else 0.0),
        ),
    )
    verdict = "Likely novel / OOD" if ood_score >= OOD_THRESHOLD else "Likely in-distribution"

    pool = trainer.system.ccc_pool
    pool_stats = pool.get_pool_stats()
    fired_rows = []
    for position, ccc_index in enumerate(step_result.fired_indices):
        confidence = (
            float(step_result.winner_confidences[position].item())
            if position < step_result.winner_confidences.numel()
            else float(step_result.confidence)
        )
        fired_rows.append(
            {
                "ccc_id": int(ccc_index),
                "confidence": confidence,
                "label": _dominant_label_for_ccc(trainer, label_names, int(ccc_index)),
            }
        )
    fired_rows.sort(key=lambda row: row["confidence"], reverse=True)

    locked_cccs = _count_locked_cccs(pool)
    committed_cccs = int(pool_stats["num_committed"])
    active_cccs = len(step_result.fired_indices)
    precision = float(getattr(pool, "get_precision", lambda: 1.0)())
    summary = (
        f"**Prediction:** {'I don’t know' if abstained else prediction_name}  \n"
        f"**Top confidence:** {top_confidence:.3f}  \n"
        f"**OOD verdict:** {verdict}  \n"
        f"**Pool precision:** {precision:.3f}"
    )
    return {
        "label_distribution": label_distribution,
        "prediction_name": prediction_name,
        "prediction_confidence": top_confidence,
        "abstained": abstained,
        "ccc_rows": fired_rows,
        "ccc_html": _ccc_visual_html(
            fired_rows[:8],
            active_cccs=active_cccs,
            locked_cccs=locked_cccs,
            committed_cccs=committed_cccs,
        ),
        "pool_stats": {
            "source": pool.__class__.__name__,
            "data_source": getattr(trainer, "_demo_source", "cached-demo-model"),
            "committed_cccs": committed_cccs,
            "locked_cccs": locked_cccs,
            "active_cccs_this_inference": active_cccs,
            "num_uncommitted": int(pool_stats["num_uncommitted"]),
            "fire_rate": float(pool_stats["fire_rate"]),
            "mean_confidence": float(pool_stats["mean_confidence"]),
            "mean_importance": float(pool_stats["mean_importance"]),
        },
        "ood_score": float(ood_score),
        "precision": precision,
        "locked_cccs": locked_cccs,
        "active_cccs": active_cccs,
        "summary": summary,
    }


def _classify_image(
    image: Any,
    *,
    trainer: VisionTrainer,
    class_centroids: dict[int, torch.Tensor],
    label_examples: dict[int, torch.Tensor],
    label_names: dict[int, str],
) -> dict[str, Any]:
    tensor = _coerce_image_tensor(image)
    return _classify_tensor(
        trainer,
        tensor,
        class_centroids=class_centroids,
        label_examples=label_examples,
        label_names=label_names,
    )


def _base_label_names() -> dict[int, str]:
    return {digit: str(digit) for digit in range(10)}


def _pool_snapshot(trainer: VisionTrainer) -> dict[str, Any]:
    pool_stats = trainer.system.ccc_pool.get_pool_stats()
    return {
        "committed_cccs": int(pool_stats["num_committed"]),
        "locked_cccs": _count_locked_cccs(trainer.system.ccc_pool),
        "available_cccs": int(pool_stats["num_uncommitted"]),
        "fire_rate": float(pool_stats["fire_rate"]),
        "mean_confidence": float(pool_stats["mean_confidence"]),
        "precision": float(getattr(trainer.system.ccc_pool, "get_precision", lambda: 1.0)()),
    }


def _build_online_state() -> OnlineLearningState:
    backend = get_demo_backend()
    trainer = _clone_with_torch(backend.trainer)
    setattr(trainer, "_demo_source", backend.source)
    return OnlineLearningState(
        trainer=trainer,
        label_names=_base_label_names(),
        label_to_id={str(digit): digit for digit in range(10)},
        class_centroids={label: centroid.clone() for label, centroid in backend.class_centroids.items()},
        class_counts=dict(backend.class_counts),
        label_examples={label: example.clone() for label, example in backend.label_examples.items()},
        events=[],
    )


def _ensure_online_state(state: OnlineLearningState | None) -> OnlineLearningState:
    return state if state is not None else _build_online_state()


def _update_label_centroid(state: OnlineLearningState, label_id: int, tensor: torch.Tensor) -> None:
    flat = tensor.reshape(-1).to(torch.float32)
    if label_id not in state.class_centroids:
        state.class_centroids[label_id] = _normalize_vector(flat)
        state.class_counts[label_id] = 1
        state.label_examples[label_id] = tensor.clone()
        return
    count = max(int(state.class_counts.get(label_id, 0)), 1)
    updated = (state.class_centroids[label_id] * count) + _normalize_vector(flat)
    state.class_counts[label_id] = count + 1
    state.class_centroids[label_id] = _normalize_vector(updated / float(count + 1))
    state.label_examples[label_id] = tensor.clone()


def _committed_ccc_indices(trainer: VisionTrainer) -> set[int]:
    indices: set[int] = set()
    cccs = getattr(trainer.system.ccc_pool, "cccs", None)
    if cccs is None:
        return indices
    for index, ccc in enumerate(cccs):
        if bool(getattr(ccc, "is_committed").item()):
            indices.add(index)
    return indices


def _force_single_example_recruitment(
    trainer: VisionTrainer,
    tensor: torch.Tensor,
    label_id: int,
) -> tuple[int | None, bool, bool, float, Any]:
    cccs = getattr(trainer.system.ccc_pool, "cccs", None)
    if cccs is None:
        return trainer._train_single_sample(tensor, label_id)  # noqa: SLF001
    saved_thresholds: list[tuple[Any, float]] = []
    for ccc in cccs:
        if bool(getattr(ccc, "is_committed").item()):
            theta = ccc.margin_gate.theta_margin
            saved_thresholds.append((theta, float(theta.item())))
            theta.fill_(max(float(theta.item()), 0.98))
    try:
        return trainer._train_single_sample(tensor, label_id)  # noqa: SLF001
    finally:
        for theta, value in saved_thresholds:
            theta.fill_(value)


def classify_demo(drawn_image: Any, uploaded_image: Any) -> tuple[dict[str, float], str, dict[str, Any], float, float, float, float, str]:
    backend = get_demo_backend()
    setattr(backend.trainer, "_demo_source", backend.source)
    report = _classify_image(
        _pick_image(drawn_image, uploaded_image),
        trainer=backend.trainer,
        class_centroids=backend.class_centroids,
        label_examples=backend.label_examples,
        label_names=_base_label_names(),
    )
    return (
        report["label_distribution"],
        report["ccc_html"],
        report["pool_stats"],
        report["ood_score"],
        report["precision"],
        float(report["locked_cccs"]),
        float(report["active_cccs"]),
        report["summary"],
    )


def reset_online_learning() -> tuple[OnlineLearningState, str, dict[str, Any], dict[str, Any], str, dict[str, float], str, dict[str, Any]]:
    state = _build_online_state()
    return (
        state,
        "Online session reset. Teach a novel symbol such as `x-shape` or `triangle` to recruit a fresh CCC.",
        {},
        {},
        "No learning events yet.",
        {},
        "",
        {},
    )


def teach_online_learning(
    state: OnlineLearningState | None,
    drawn_image: Any,
    uploaded_image: Any,
    label_text: str,
) -> tuple[OnlineLearningState, str, dict[str, Any], dict[str, Any], str, dict[str, float], str, dict[str, Any]]:
    state = _ensure_online_state(state)
    clean_label = (label_text or "").strip().lower()
    if not clean_label:
        raise gr.Error("Enter a label before teaching the new concept.")

    image = _pick_image(drawn_image, uploaded_image)
    tensor = _coerce_image_tensor(image)
    before_stats = _pool_snapshot(state.trainer)
    before_indices = _committed_ccc_indices(state.trainer)

    is_new_label = clean_label not in state.label_to_id
    if not is_new_label:
        label_id = int(state.label_to_id[clean_label])
    else:
        label_id = max(state.label_names, default=-1) + 1
        state.label_to_id[clean_label] = label_id
        state.label_names[label_id] = clean_label

    prediction, abstained, recruited, confidence, step_result = state.trainer._train_single_sample(  # noqa: SLF001
        tensor,
        label_id,
    )
    if is_new_label:
        interim_indices = _committed_ccc_indices(state.trainer)
        if not (interim_indices - before_indices):
            prediction, abstained, recruited, confidence, step_result = _force_single_example_recruitment(
                state.trainer,
                tensor,
                label_id,
            )
    _update_label_centroid(state, label_id, tensor)

    after_stats = _pool_snapshot(state.trainer)
    after_indices = _committed_ccc_indices(state.trainer)
    new_indices = sorted(after_indices - before_indices)
    report = _classify_tensor(
        state.trainer,
        tensor,
        class_centroids=state.class_centroids,
        label_examples=state.label_examples,
        label_names=state.label_names,
    )

    event = {
        "label": clean_label,
        "new_ccc_indices": new_indices,
        "recruited": bool(recruited or new_indices),
        "confidence": float(confidence),
        "prediction_before_update": None if prediction is None else state.label_names.get(int(prediction), str(prediction)),
        "abstained_before_update": bool(abstained),
        "fired_cccs": [int(index) for index in step_result.fired_indices],
    }
    state.events.append(event)
    state.events = state.events[-6:]

    if event["recruited"]:
        recruit_text = ", ".join(f"CCC {index}" for index in new_indices) if new_indices else "a fresh CCC path"
        status = f"Learned `{clean_label}` with {recruit_text}."
    else:
        status = f"Updated `{clean_label}` using existing CCCs. Try a more novel sketch to force recruitment."

    event_markdown = (
        f"**Label:** `{clean_label}`  \n"
        f"**Recruitment:** {'yes' if event['recruited'] else 'no'}  \n"
        f"**Fired CCCs:** {', '.join(str(index) for index in event['fired_cccs']) or 'none'}  \n"
        f"**Post-teach prediction:** {'I don’t know' if report['abstained'] else report['prediction_name']}"
    )
    return (
        state,
        status,
        before_stats,
        after_stats,
        event_markdown,
        report["label_distribution"],
        report["ccc_html"],
        {"recent_events": state.events},
    )


def classify_online_learning(
    state: OnlineLearningState | None,
    drawn_image: Any,
    uploaded_image: Any,
) -> tuple[OnlineLearningState, dict[str, float], str, dict[str, Any], str]:
    state = _ensure_online_state(state)
    report = _classify_image(
        _pick_image(drawn_image, uploaded_image),
        trainer=state.trainer,
        class_centroids=state.class_centroids,
        label_examples=state.label_examples,
        label_names=state.label_names,
    )
    return (
        state,
        report["label_distribution"],
        report["ccc_html"],
        report["pool_stats"],
        report["summary"],
    )


def _subset_by_labels(samples: list[Sample], labels: set[int], *, per_label: int) -> list[Sample]:
    counts: Counter[int] = Counter()
    selected: list[Sample] = []
    for tensor, label in samples:
        if label not in labels or counts[label] >= per_label:
            continue
        selected.append((tensor.clone(), int(label)))
        counts[label] += 1
        if all(counts[digit] >= per_label for digit in labels):
            return selected
    return selected


def _build_continual_trainer() -> VisionTrainer:
    config = VisionTrainConfig(
        input_dim=28 * 28,
        concept_dim=64,
        max_pool_size=48,
        margin_threshold=0.5,
        use_batched=False,
        batch_size=16,
        learning_rate=0.04,
        num_train_samples=CONTINUAL_TRAIN_PER_CLASS * 5,
        num_test_samples=CONTINUAL_TEST_PER_CLASS * 5,
        preprocessing_warmup_samples=0,
        freeze_f1_after=0,
        consolidation_strength=0.0,
    )
    return VisionTrainer(config)


def _lock_committed_concepts(trainer: VisionTrainer) -> None:
    cccs = getattr(trainer.system.ccc_pool, "cccs", None)
    if cccs is None:
        return
    for ccc in cccs:
        if bool(getattr(ccc, "is_committed").item()) and hasattr(ccc, "locked"):
            ccc.locked.fill_(True)


def _build_continual_state() -> ContinualDemoState:
    backend = get_demo_backend()
    task1_labels = {0, 1, 2, 3, 4}
    task2_labels = {5, 6, 7, 8, 9}
    task_train = [
        _subset_by_labels(backend.train_samples, task1_labels, per_label=CONTINUAL_TRAIN_PER_CLASS),
        _subset_by_labels(backend.train_samples, task2_labels, per_label=CONTINUAL_TRAIN_PER_CLASS),
    ]
    task_test = [
        _subset_by_labels(backend.test_samples, task1_labels, per_label=CONTINUAL_TEST_PER_CLASS),
        _subset_by_labels(backend.test_samples, task2_labels, per_label=CONTINUAL_TEST_PER_CLASS),
    ]
    trainers = {
        "without locking": _build_continual_trainer(),
        "with locking": _build_continual_trainer(),
    }
    return ContinualDemoState(
        trainers=trainers,
        stage_index=0,
        task_train=task_train,
        task_test=task_test,
        history={"without locking": [], "with locking": []},
        source=backend.source,
    )


def _ensure_continual_state(state: ContinualDemoState | None) -> ContinualDemoState:
    return state if state is not None else _build_continual_state()


def _evaluate_continual_row(trainer: VisionTrainer, task_test: list[list[Sample]]) -> list[float]:
    row: list[float] = []
    for samples in task_test:
        metrics = trainer.evaluate(samples, num_samples=len(samples))
        row.append(float(metrics["accuracy"]))
    return row


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _final_bwt(rows: list[list[float]]) -> float:
    if len(rows) < 2:
        return 0.0
    return float(rows[-1][0] - rows[0][0])


def _continual_table_rows(state: ContinualDemoState) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, history in state.history.items():
        trainer = state.trainers[name]
        for stage_number, row in enumerate(history, start=1):
            task2_value = row[1] if stage_number > 1 else None
            rows.append(
                [
                    name,
                    f"After task {stage_number}",
                    _format_percent(row[0]),
                    _format_percent(task2_value),
                    f"{_final_bwt(history) * 100:+.1f}%",
                    str(_count_locked_cccs(trainer.system.ccc_pool)),
                    str(int(trainer.system.ccc_pool.get_pool_stats()["num_committed"])),
                ]
            )
    return rows


def _continual_json(state: ContinualDemoState) -> dict[str, Any]:
    output: dict[str, Any] = {"source": state.source, "stage": state.stage_index}
    for name, history in state.history.items():
        output[name] = {
            "evaluation_matrix": history,
            "bwt": _final_bwt(history),
            "locked_cccs": _count_locked_cccs(state.trainers[name].system.ccc_pool),
        }
    return output


def _continual_summary(state: ContinualDemoState) -> str:
    if state.stage_index == 0:
        return (
            f"Dataset source: **{state.source}**. Train Task 1 to establish digits 0–4, "
            "then Task 2 to measure backward transfer."
        )
    if state.stage_index == 1:
        return (
            "Task 1 complete. The `with locking` run has already frozen its committed CCCs. "
            "Train Task 2, then press **Evaluate All** to compare BWT."
        )
    without_bwt = _final_bwt(state.history["without locking"])
    with_bwt = _final_bwt(state.history["with locking"])
    delta = with_bwt - without_bwt
    if delta > 0:
        headline = f"Concept locking improved BWT by {delta * 100:.1f} points."
    elif delta < 0:
        headline = (
            f"On this small {state.source.lower()} stream, locking trails by {abs(delta) * 100:.1f} points; "
            "the gap can be small on synthetic fallback data."
        )
    else:
        headline = "Both runs landed on the same BWT on this small stream."
    return (
        f"{headline}  \n"
        f"- Without locking BWT: {without_bwt * 100:+.1f}%  \n"
        f"- With locking BWT: {with_bwt * 100:+.1f}%"
    )


def reset_continual_demo() -> tuple[ContinualDemoState, str, list[list[str]], dict[str, Any]]:
    state = _build_continual_state()
    return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)


def train_task_1(state: ContinualDemoState | None) -> tuple[ContinualDemoState, str, list[list[str]], dict[str, Any]]:
    state = _ensure_continual_state(state)
    if state.stage_index >= 1:
        return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)

    for name, trainer in state.trainers.items():
        trainer.train_online(
            state.task_train[0],
            num_samples=len(state.task_train[0]),
            interleave_classes=True,
        )
        if name == "with locking":
            _lock_committed_concepts(trainer)
        state.history[name].append(_evaluate_continual_row(trainer, state.task_test))
    state.stage_index = 1
    return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)


def train_task_2(state: ContinualDemoState | None) -> tuple[ContinualDemoState, str, list[list[str]], dict[str, Any]]:
    state = _ensure_continual_state(state)
    if state.stage_index == 0:
        state, _, _, _ = train_task_1(state)
    if state.stage_index >= 2:
        return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)

    for name, trainer in state.trainers.items():
        trainer.train_online(
            state.task_train[1],
            num_samples=len(state.task_train[1]),
            interleave_classes=True,
        )
        state.history[name].append(_evaluate_continual_row(trainer, state.task_test))
    state.stage_index = 2
    return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)


def evaluate_all_tasks(state: ContinualDemoState | None) -> tuple[ContinualDemoState, str, list[list[str]], dict[str, Any]]:
    state = _ensure_continual_state(state)
    return state, _continual_summary(state), _continual_table_rows(state), _continual_json(state)


def _overview_markdown(backend: DemoBackend) -> str:
    pool_stats = backend.trainer.system.ccc_pool.get_pool_stats()
    return (
        "# Bio-ARN 2.0 Gradio Demo\n\n"
        "CPU-only conference demo with cached MNIST training, CCC introspection, online recruitment, "
        "and a small continual-learning walkthrough.\n\n"
        f"- **Dataset source:** {backend.source}\n"
        f"- **Cached model:** `{MODEL_CACHE_PATH.relative_to(PROJECT_ROOT)}`\n"
        f"- **Committed CCCs:** {int(pool_stats['num_committed'])}\n"
        f"- **Eval accuracy (raw Bio-ARN readout):** {float(backend.eval_summary['accuracy']) * 100:.1f}%\n"
    )


def _architecture_markdown() -> str:
    return (
        "## Architecture and headline metrics\n\n"
        "- **278× projected energy efficiency** versus dense GPU inference\n"
        "- **1.000 OOD AUROC** on the project benchmark suite\n"
        "- **Core ideas:** concept cell clusters (CCCs), honest abstention, precision-weighted learning, and concept locking\n\n"
        f"[Paper / architecture notes]({README_PAPER_URL})  \n"
        f"[GitHub repository]({README_GITHUB_URL})"
    )


def create_app(*, load_backend: bool = True) -> gr.Blocks:
    backend = get_demo_backend() if load_backend else None
    examples = (
        [[_tensor_to_uint8_image(backend.label_examples[digit])] for digit in (0, 4, 9)]
        if backend is not None
        else []
    )
    with gr.Blocks(title="Bio-ARN 2.0 Demo") as demo:
        gr.Markdown(_overview_markdown(backend) if backend is not None else "# Bio-ARN 2.0 Gradio Demo")

        with gr.Tabs():
            with gr.Tab("Image Classification"):
                gr.Markdown(
                    "Upload or draw a digit-like image. The panel shows the class distribution, fired CCCs, "
                    "OOD score, precision signal, and concept locking state."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        classify_draw = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw a digit or simple symbol",
                            height=280,
                            width=280,
                        )
                        classify_upload = gr.Image(
                            image_mode="L",
                            type="numpy",
                            label="...or upload an image",
                            height=280,
                        )
                        if examples:
                            gr.Examples(examples=examples, inputs=[classify_upload], label="Quick examples")
                        classify_button = gr.Button("Classify", variant="primary")
                    with gr.Column(scale=1):
                        classify_output = gr.Label(label="Prediction", num_top_classes=5)
                        classify_summary = gr.Markdown()
                        ccc_html = gr.HTML(label="CCC pool")
                        with gr.Row():
                            ood_value = gr.Number(label="OOD score", precision=3)
                            precision_value = gr.Number(label="Precision signal", precision=3)
                        with gr.Row():
                            locked_value = gr.Number(label="Locked CCCs", precision=0)
                            active_value = gr.Number(label="Active CCCs", precision=0)
                        pool_json = gr.JSON(label="Pool stats")

                classify_button.click(
                    classify_demo,
                    inputs=[classify_draw, classify_upload],
                    outputs=[
                        classify_output,
                        ccc_html,
                        pool_json,
                        ood_value,
                        precision_value,
                        locked_value,
                        active_value,
                        classify_summary,
                    ],
                )

            with gr.Tab("Online Learning"):
                gr.Markdown(
                    "Teach Bio-ARN a brand-new visual concept with one labelled example, then classify similar "
                    "images and watch the pool reuse or recruit CCCs."
                )
                online_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column():
                        teach_draw = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw a novel concept",
                            height=280,
                            width=280,
                        )
                        teach_upload = gr.Image(
                            image_mode="L",
                            type="numpy",
                            label="...or upload one example",
                            height=280,
                        )
                        teach_label = gr.Textbox(
                            label="Label",
                            placeholder="Examples: x-shape, triangle, my-logo",
                        )
                        with gr.Row():
                            teach_button = gr.Button("Recruit CCC", variant="primary")
                            reset_online_button = gr.Button("Reset online session")
                    with gr.Column():
                        teach_status = gr.Markdown("No learning events yet.")
                        before_stats = gr.JSON(label="Before pool stats")
                        after_stats = gr.JSON(label="After pool stats")
                        teach_event = gr.Markdown()
                        teach_result = gr.Label(label="Immediate post-teach classification", num_top_classes=5)
                        teach_ccc_html = gr.HTML(label="Recruitment view")
                        teach_events_json = gr.JSON(label="Recent learning events")

                with gr.Row():
                    with gr.Column():
                        probe_draw = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw a similar follow-up image",
                            height=220,
                            width=220,
                        )
                        probe_upload = gr.Image(
                            image_mode="L",
                            type="numpy",
                            label="...or upload a similar image",
                            height=220,
                        )
                        probe_button = gr.Button("Classify with updated pool")
                    with gr.Column():
                        probe_output = gr.Label(label="Updated prediction", num_top_classes=5)
                        probe_ccc_html = gr.HTML()
                        probe_pool_json = gr.JSON(label="Updated pool stats")
                        probe_summary = gr.Markdown()

                teach_button.click(
                    teach_online_learning,
                    inputs=[online_state, teach_draw, teach_upload, teach_label],
                    outputs=[
                        online_state,
                        teach_status,
                        before_stats,
                        after_stats,
                        teach_event,
                        teach_result,
                        teach_ccc_html,
                        teach_events_json,
                    ],
                )
                probe_button.click(
                    classify_online_learning,
                    inputs=[online_state, probe_draw, probe_upload],
                    outputs=[online_state, probe_output, probe_ccc_html, probe_pool_json, probe_summary],
                )
                reset_online_button.click(
                    reset_online_learning,
                    outputs=[
                        online_state,
                        teach_status,
                        before_stats,
                        after_stats,
                        teach_event,
                        teach_result,
                        teach_ccc_html,
                        teach_events_json,
                    ],
                )

            with gr.Tab("Continual Learning Demo"):
                gr.Markdown(
                    "Small two-stage split-MNIST-like walkthrough. Task 1 trains digits 0–4, Task 2 trains digits 5–9. "
                    "Compare BWT with and without concept locking."
                )
                continual_state = gr.State(value=None)
                with gr.Row():
                    train_task1_button = gr.Button("Train Task 1", variant="primary")
                    train_task2_button = gr.Button("Train Task 2")
                    eval_all_button = gr.Button("Evaluate All")
                    reset_continual_button = gr.Button("Reset demo")

                continual_summary = gr.Markdown("Train Task 1 to begin.")
                continual_table = gr.Dataframe(
                    headers=[
                        "Config",
                        "Stage",
                        "Task 1 acc",
                        "Task 2 acc",
                        "BWT",
                        "Locked CCCs",
                        "Committed CCCs",
                    ],
                    datatype=["str"] * 7,
                    interactive=False,
                    row_count=0,
                    column_count=(7, "fixed"),
                    label="With vs without locking",
                )
                continual_json = gr.JSON(label="Raw continual metrics")

                train_task1_button.click(
                    train_task_1,
                    inputs=[continual_state],
                    outputs=[continual_state, continual_summary, continual_table, continual_json],
                )
                train_task2_button.click(
                    train_task_2,
                    inputs=[continual_state],
                    outputs=[continual_state, continual_summary, continual_table, continual_json],
                )
                eval_all_button.click(
                    evaluate_all_tasks,
                    inputs=[continual_state],
                    outputs=[continual_state, continual_summary, continual_table, continual_json],
                )
                reset_continual_button.click(
                    reset_continual_demo,
                    outputs=[continual_state, continual_summary, continual_table, continual_json],
                )

            with gr.Tab("Architecture Info"):
                gr.Markdown(_architecture_markdown())
                if FIGURE_PATH.exists():
                    gr.Image(
                        value=str(FIGURE_PATH),
                        label="Bio-ARN architecture diagram",
                        interactive=False,
                    )
                else:
                    gr.Markdown("`docs/figures/figure1_full_bioarn_pipeline.png` was not found.")

    return demo


def main() -> None:
    backend = get_demo_backend()
    print(f"[demo] ready using {backend.source}")
    app = create_app(load_backend=True)
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
