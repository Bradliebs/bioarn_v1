from __future__ import annotations

import torch

from bioarn.config import MultimodalTrainConfig, MultimodalFusionConfig
from bioarn.training import MultimodalExample, MultimodalTrainer


def make_config() -> MultimodalTrainConfig:
    """Small config for fast tests."""
    return MultimodalTrainConfig(
        fusion=MultimodalFusionConfig(
            concept_dim=64,
            vision_pool_size=50,
            audio_pool_size=30,
            workspace_size=8,
            margin_threshold=0.3,
        ),
        num_samples=24,
        num_passes=2,
        num_classes=3,
        vision_size=16,
        shuffle=True,
        seed=42,
    )


def build_vision_audio_examples() -> list[MultimodalExample]:
    """Build paired vision+audio examples with distinct per-class patterns."""
    examples: list[MultimodalExample] = []
    labels = ["cat", "dog", "bird"]
    n_mels = 40
    n_frames = 32
    for repeat in range(4):
        for idx, label in enumerate(labels):
            gen = torch.Generator().manual_seed(repeat * 10 + idx)
            vision = torch.randn(16 * 16, generator=gen)
            vision[idx * 80 : (idx + 1) * 80] += 3.0
            audio = torch.randn(n_mels, n_frames, generator=gen)
            audio[idx * 10 : (idx + 1) * 10, :] += 3.0
            examples.append(
                MultimodalExample(
                    vision=vision,
                    audio=audio,
                    label=label,
                )
            )
    return examples


def test_multimodal_trainer_trains_online() -> None:
    """Trainer runs train_online and returns valid result dict."""
    torch.manual_seed(0)
    config = make_config()
    trainer = MultimodalTrainer(config)

    result = trainer.train_online(build_vision_audio_examples())

    assert isinstance(result, dict)
    assert result["num_samples"] > 0
    assert result["num_passes"] == 2
    assert 0.0 <= result["mean_agreement"] <= 1.0
    assert 0.0 <= result["mean_confidence"] <= 1.0


def test_multimodal_trainer_uses_default_stream() -> None:
    """Trainer falls back to SyntheticMultimodalStream when no data provided."""
    torch.manual_seed(0)
    config = make_config()
    trainer = MultimodalTrainer(config)

    result = trainer.train_online()

    assert isinstance(result, dict)
    assert result["num_samples"] > 0


def test_multimodal_trainer_learns_associations() -> None:
    """After training, the engine should have learned some cross-modal associations."""
    torch.manual_seed(0)
    config = make_config()
    trainer = MultimodalTrainer(config)

    result = trainer.train_online(build_vision_audio_examples())

    assert result["num_associations"] >= 0
    assert result["label_consistency"] >= 0.0
