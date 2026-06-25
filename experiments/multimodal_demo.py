"""Runnable shared-CCC multimodal training demo."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioarn.multimodal import MultimodalConfig
from bioarn.training import MultimodalExample, MultimodalTrainer


def digit_pattern(label: str) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    if label == "zero":
        image[5, 8:20] = 1.0
        image[22, 8:20] = 1.0
        image[5:23, 8] = 1.0
        image[5:23, 19] = 1.0
    elif label == "one":
        image[5:23, 14] = 1.0
        image[22, 10:18] = 1.0
    elif label == "two":
        image[6, 8:20] = 1.0
        image[14, 8:20] = 1.0
        image[22, 8:20] = 1.0
        image[7:14, 19] = 1.0
        image[14:22, 8] = 1.0
    else:
        raise ValueError(f"Unsupported label: {label}")
    return image


def noisy_variant(image: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = 0.04 * torch.randn(image.shape, generator=generator)
    return (image + noise).clamp_(0.0, 1.0)


def build_examples(repeats: int = 4) -> list[MultimodalExample]:
    labels = ["zero", "one", "two"]
    dataset: list[MultimodalExample] = []
    for repeat in range(repeats):
        for index, label in enumerate(labels):
            dataset.append(
                MultimodalExample(
                    vision=noisy_variant(digit_pattern(label), seed=(repeat * 17) + index),
                    text=label,
                    label=label,
                )
            )
    return dataset


def main() -> None:
    torch.manual_seed(0)
    trainer = MultimodalTrainer(
        MultimodalConfig(
            vision_dim=28 * 28,
            language_dim=64,
            concept_dim=64,
            cross_modal_strength=0.8,
            temporal_window=3,
            alignment_threshold=0.45,
        )
    )

    examples = build_examples()
    train_result = trainer.train(examples, epochs=2)
    eval_result = trainer.evaluate(examples)

    print("=== Bio-ARN Multimodal Demo ===")
    print(f"pairs: {train_result.num_pairs}")
    print(f"steps: {train_result.num_steps}")
    print(f"committed_cccs: {train_result.committed_cccs}")
    print(f"shared_cccs: {train_result.shared_cccs}")
    print(f"sharing_ratio: {train_result.concept_sharing_ratio:.3f}")
    print(f"retrieval_accuracy: {eval_result.cross_modal_retrieval_accuracy:.3f}")
    print(f"mean_association_strength: {eval_result.mean_association_strength:.3f}")
    print()

    for label in ("zero", "one", "two"):
        matches = trainer.fusion.cross_modal_retrieval(digit_pattern(label), "vision", "text")
        top_label = matches[0].label if matches else "none"
        recalled = trainer.fusion.visualize_text(label)
        print(
            f"[retrieve] image={label} text={top_label} "
            f"recalled_pixels={int(torch.count_nonzero(recalled).item())}"
        )


if __name__ == "__main__":
    main()
