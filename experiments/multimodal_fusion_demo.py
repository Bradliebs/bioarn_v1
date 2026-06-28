"""Demo for GNW-based cross-modal fusion and recall."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioarn.config import (  # noqa: E402
    AudioConfig,
    AudioHierarchyConfig,
    MultimodalFusionConfig,
    MultimodalTrainConfig,
    TemporalConfig,
)
from bioarn.data.multimodal import SyntheticMultimodalStream  # noqa: E402
from bioarn.multimodal import MultimodalInput  # noqa: E402
from bioarn.training import MultimodalTrainer  # noqa: E402


def make_train_config() -> MultimodalTrainConfig:
    return MultimodalTrainConfig(
        fusion=MultimodalFusionConfig(
            concept_dim=64,
            vision_pool_size=32,
            audio_pool_size=32,
            workspace_size=4,
            cross_modal_weight=0.55,
            agreement_threshold=0.55,
            margin_threshold=0.2,
            learning_rate=0.08,
            audio=AudioConfig(
                sample_rate=4000,
                n_mels=24,
                n_fft=128,
                hop_length=40,
                max_duration_ms=240,
            ),
            audio_hierarchy=AudioHierarchyConfig(
                n_mels=24,
                a1_channels=8,
                a2_channels=12,
                belt_dim=48,
                temporal_kernel=5,
                init_seed=11,
            ),
            temporal=TemporalConfig(
                context_window=4,
                concept_dim=64,
                stdp_tau_plus=8.0,
                stdp_tau_minus=16.0,
                stdp_lr=0.2,
                prediction_threshold=0.15,
            ),
        ),
        num_samples=32,
        num_passes=2,
        num_classes=4,
        vision_size=16,
        shuffle=True,
        seed=8,
        learning_rate_multiplier=1.0,
    )


def main() -> None:
    torch.manual_seed(0)
    config = make_train_config()
    stream = SyntheticMultimodalStream(
        config.num_samples,
        num_classes=config.num_classes,
        image_size=config.vision_size,
        sample_rate=config.fusion.audio.sample_rate,
        duration_ms=config.fusion.audio.max_duration_ms,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    trainer = MultimodalTrainer(config)
    metrics = trainer.train_online(stream)

    probe = stream.sample_for_label("diagonal", variant=91)
    audio_only = trainer.engine.process(MultimodalInput(audio=probe.audio, metadata=probe.metadata))
    vision_only = trainer.engine.process(MultimodalInput(vision=probe.vision, metadata=probe.metadata))
    similarity = float(
        F.cosine_similarity(
            audio_only.concept_direction.unsqueeze(0).to(vision_only.concept_direction),
            vision_only.concept_direction.unsqueeze(0),
        ).item()
    )

    recalled_visual = audio_only.per_modality.get("vision")
    recalled_audio = vision_only.per_modality.get("audio")

    print("=== Bio-ARN Multimodal Fusion Demo ===")
    print(f"samples: {metrics['num_samples']} x {metrics['num_passes']} passes")
    print(f"label consistency: {metrics['label_consistency']:.3f}")
    print(f"mean agreement: {metrics['mean_agreement']:.3f}")
    print(f"associations: {metrics['num_associations']}")
    print()
    print("Cross-modal binding probe: diagonal")
    print(
        "audio-only -> "
        f"winner={audio_only.winner_modality}, "
        f"visual recall={None if recalled_visual is None else recalled_visual.fired_indices}, "
        f"precision={audio_only.precision:.3f}"
    )
    print(
        "vision-only -> "
        f"winner={vision_only.winner_modality}, "
        f"audio recall={None if recalled_audio is None else recalled_audio.fired_indices}, "
        f"precision={vision_only.precision:.3f}"
    )
    print(f"shared concept cosine: {similarity:.3f}")


if __name__ == "__main__":
    main()
