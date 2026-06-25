from __future__ import annotations

import torch

from bioarn.multimodal import MultimodalConfig
from bioarn.training import MultimodalExample, MultimodalTrainer, MultimodalTrainingResult


def make_config() -> MultimodalConfig:
    return MultimodalConfig(
        vision_dim=28 * 28,
        language_dim=64,
        concept_dim=64,
        cross_modal_strength=0.8,
        temporal_window=3,
        max_description_length=12,
        alignment_threshold=0.45,
    )


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
    noise = 0.03 * torch.randn(image.shape, generator=generator)
    return (image + noise).clamp_(0.0, 1.0)


def build_examples() -> list[MultimodalExample]:
    labels = ["zero", "one", "two"]
    dataset: list[MultimodalExample] = []
    for repeat in range(2):
        for index, label in enumerate(labels):
            dataset.append(
                MultimodalExample(
                    vision=noisy_variant(digit_pattern(label), seed=(repeat * 11) + index),
                    text=label,
                    label=label,
                )
            )
    return dataset


def test_multimodal_trainer_shares_cccs() -> None:
    torch.manual_seed(0)
    trainer = MultimodalTrainer(make_config())

    result = trainer.train(build_examples(), epochs=2)

    assert isinstance(result, MultimodalTrainingResult)
    for label in ("zero", "one", "two"):
        assert trainer.fusion.label_to_ccc[("vision", label)] == trainer.fusion.label_to_ccc[("text", label)]
    assert result.shared_cccs >= 3
    assert result.concept_sharing_ratio > 0.0


def test_multimodal_trainer_interleaves_modalities() -> None:
    torch.manual_seed(0)
    trainer = MultimodalTrainer(make_config())

    result = trainer.train(build_examples(), epochs=1)

    assert result.num_steps == result.num_pairs * 2
    assert result.modality_sequence == ("vision", "text") * result.num_pairs
    assert result.converged_pairs == result.num_pairs


def test_multimodal_trainer_evaluate_reports_retrieval_metrics() -> None:
    torch.manual_seed(0)
    trainer = MultimodalTrainer(make_config())
    examples = build_examples()
    trainer.train(examples, epochs=2)

    evaluation = trainer.evaluate(
        [
            MultimodalExample(vision=noisy_variant(example.vision, seed=index + 101), text=example.text, label=example.label)
            for index, example in enumerate(examples[:3])
        ]
    )

    assert evaluation.cross_modal_retrieval_accuracy >= 0.9
    assert evaluation.mean_association_strength > 0.0
    assert evaluation.mean_reciprocal_rank >= evaluation.cross_modal_retrieval_accuracy
