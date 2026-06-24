"""Demo: Bio-ARN processing vision and language together."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bioarn.data.base import DataSample
from bioarn.multimodal import ModalityAligner, MultimodalConfig, MultimodalFusion


@dataclass(frozen=True)
class DemoResult:
    label: str
    predicted: str
    strength: float


def make_pattern(kind: str) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    if kind == "horizontal_top":
        image[6, 4:24] = 1.0
    elif kind == "horizontal_mid":
        image[14, 4:24] = 1.0
    elif kind == "vertical_left":
        image[4:24, 8] = 1.0
    elif kind == "vertical_mid":
        image[4:24, 14] = 1.0
    elif kind == "diagonal":
        idx = torch.arange(5, 23)
        image[idx, idx] = 1.0
    elif kind == "anti_diagonal":
        idx = torch.arange(5, 23)
        image[idx, 27 - idx] = 1.0
    elif kind == "cross":
        image[14, 4:24] = 1.0
        image[4:24, 14] = 1.0
    elif kind == "x_shape":
        idx = torch.arange(5, 23)
        image[idx, idx] = 1.0
        image[idx, 27 - idx] = 1.0
    elif kind == "box":
        image[6, 6:22] = 1.0
        image[21, 6:22] = 1.0
        image[6:22, 6] = 1.0
        image[6:22, 21] = 1.0
    elif kind == "center_dot":
        image[13:15, 13:15] = 1.0
    else:
        raise ValueError(f"Unknown pattern: {kind}")
    return image


def noisy_variant(base: torch.Tensor, seed: int, noise: float = 0.04) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (base + noise * torch.randn(base.shape, generator=generator)).clamp_(0.0, 1.0)


def build_catalog() -> dict[str, torch.Tensor]:
    labels = [
        "horizontal_top",
        "horizontal_mid",
        "vertical_left",
        "vertical_mid",
        "diagonal",
        "anti_diagonal",
        "cross",
        "x_shape",
        "box",
        "center_dot",
    ]
    return {label: make_pattern(label) for label in labels}


def evaluate_supervised(fusion: MultimodalFusion, catalog: dict[str, torch.Tensor]) -> tuple[list[DemoResult], float]:
    results: list[DemoResult] = []
    correct = 0
    for index, (label, image) in enumerate(catalog.items()):
        query = noisy_variant(image, seed=100 + index)
        matches = fusion.cross_modal_retrieval(query, "vision", "text")
        predicted = matches[0].label if matches else "<none>"
        strength = matches[0].strength if matches else 0.0
        results.append(DemoResult(label=label, predicted=predicted or "<none>", strength=float(strength)))
        correct += int(predicted == label)
    accuracy = correct / max(1, len(catalog))
    return results, accuracy


def main() -> None:
    torch.manual_seed(0)
    catalog = build_catalog()
    config = MultimodalConfig(
        vision_dim=28 * 28,
        language_dim=64,
        concept_dim=64,
        cross_modal_strength=0.8,
        temporal_window=3,
        max_description_length=12,
        alignment_threshold=0.55,
    )

    fusion = MultimodalFusion(config)
    aligner = ModalityAligner(fusion)

    train_images: list[torch.Tensor] = []
    train_texts: list[str] = []
    train_labels: list[str] = []
    for label, image in catalog.items():
        for seed in range(3):
            train_images.append(noisy_variant(image, seed=seed))
            train_texts.append(label)
            train_labels.append(label)
    aligner.align_by_label(train_images, train_texts, train_labels)

    supervised_results, supervised_accuracy = evaluate_supervised(fusion, catalog)
    metrics = aligner.measure_alignment([(noisy_variant(image, seed=200 + idx), label, label) for idx, (label, image) in enumerate(catalog.items())])

    sample_label = "diagonal"
    generated_label = fusion.describe_image(noisy_variant(catalog[sample_label], seed=999))
    retrieved_visual = fusion.visualize_text("vertical_mid")
    visual_similarity = float(
        torch.nn.functional.cosine_similarity(
            retrieved_visual.reshape(1, -1),
            catalog["vertical_mid"].reshape(1, -1),
        ).item()
    )

    novel_label = "box"
    novel_prediction = fusion.describe_image(noisy_variant(catalog[novel_label], seed=1234))

    temporal_fusion = MultimodalFusion(config)
    temporal_aligner = ModalityAligner(temporal_fusion)
    temporal_stream: list[DataSample] = []
    for label in ("horizontal_top", "vertical_mid", "cross"):
        temporal_stream.append(DataSample(data=noisy_variant(catalog[label], seed=300 + len(temporal_stream)), label=None, modality="vision"))
        temporal_stream.append(
            DataSample(
                data=torch.tensor(temporal_fusion.tokenizer.encode(label), dtype=torch.long),
                label=None,
                modality="text",
                metadata={"text": label},
            )
        )
    temporal_bindings = temporal_aligner.align_by_co_occurrence(temporal_stream)
    temporal_matches = temporal_fusion.cross_modal_retrieval(catalog["cross"], "vision", "text")
    temporal_strength = temporal_matches[0].strength if temporal_matches else 0.0
    temporal_prediction = temporal_matches[0].label if temporal_matches else "<none>"

    print("=== Bio-ARN Multimodal Demo ===")
    print(f"trained_categories: {len(catalog)}")
    print(f"supervised_retrieval_accuracy: {supervised_accuracy:.2%}")
    print(f"alignment_mrr: {metrics.mean_reciprocal_rank:.3f}")
    print(f"alignment_mean_strength: {metrics.mean_association_strength:.3f}")
    print(f"image_to_text: {sample_label} -> {generated_label}")
    print(f"text_to_image_similarity(vertical_mid): {visual_similarity:.3f}")
    print(f"novel_noisy_variant: {novel_label} -> {novel_prediction}")
    print(f"temporal_bindings_formed: {len(temporal_bindings)}")
    print(f"temporal_retrieval: cross -> {temporal_prediction} (strength={temporal_strength:.3f})")
    print("per_category_results:")
    for result in supervised_results:
        print(f"  {result.label:>16} -> {result.predicted:<16} strength={result.strength:.3f}")


if __name__ == "__main__":
    main()
