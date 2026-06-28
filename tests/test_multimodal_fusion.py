from __future__ import annotations

import torch
import torch.nn.functional as F

from bioarn.config import (
    AudioConfig,
    AudioHierarchyConfig,
    MultimodalFusionConfig,
    MultimodalTrainConfig,
    TemporalConfig,
)
from bioarn.data.multimodal import SyntheticMultimodalStream
from bioarn.multimodal import MultimodalFusionEngine, MultimodalInput
from bioarn.training import MultimodalTrainer


def make_fusion_config() -> MultimodalFusionConfig:
    return MultimodalFusionConfig(
        concept_dim=64,
        vision_pool_size=24,
        audio_pool_size=24,
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
    )


def make_train_config() -> MultimodalTrainConfig:
    return MultimodalTrainConfig(
        fusion=make_fusion_config(),
        num_samples=24,
        num_passes=2,
        num_classes=4,
        vision_size=16,
        shuffle=True,
        seed=5,
        learning_rate_multiplier=1.0,
    )


def train_engine(engine: MultimodalFusionEngine, stream: SyntheticMultimodalStream) -> None:
    for sample in stream.stream():
        engine.learn(sample)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0).to(right), right.unsqueeze(0)).item())


def test_process_handles_missing_modalities_gracefully() -> None:
    torch.manual_seed(0)
    engine = MultimodalFusionEngine(make_fusion_config())
    stream = SyntheticMultimodalStream(
        4,
        num_classes=2,
        image_size=16,
        sample_rate=4000,
        duration_ms=240,
        shuffle=False,
        seed=1,
    )
    sample = stream.sample_for_label("horizontal", variant=3)

    output = engine.process(MultimodalInput(vision=sample.vision, metadata=sample.metadata))

    assert output.winner_modality == "vision"
    assert output.per_modality["vision"].concept_direction is not None
    assert output.confidence >= 0.0
    assert 0.0 <= output.precision <= 1.0


def test_cross_modal_learning_forms_associations_and_broadcast_recall() -> None:
    torch.manual_seed(0)
    engine = MultimodalFusionEngine(make_fusion_config())
    stream = SyntheticMultimodalStream(
        20,
        num_classes=4,
        image_size=16,
        sample_rate=4000,
        duration_ms=240,
        shuffle=True,
        seed=2,
    )
    train_engine(engine, stream)
    probe = stream.sample_for_label("diagonal", variant=99)

    output = engine.process(MultimodalInput(audio=probe.audio, metadata=probe.metadata))

    assert output.winner_modality == "audio"
    assert "vision" in output.per_modality
    assert output.per_modality["vision"].concept_direction is not None
    assert int(engine.stats["associations"]["num_associations"]) > 0


def test_paired_modalities_converge_to_consistent_concepts() -> None:
    torch.manual_seed(0)
    engine = MultimodalFusionEngine(make_fusion_config())
    stream = SyntheticMultimodalStream(
        24,
        num_classes=4,
        image_size=16,
        sample_rate=4000,
        duration_ms=240,
        shuffle=True,
        seed=3,
    )
    train_engine(engine, stream)
    probe = stream.sample_for_label("box", variant=77)

    vision_output = engine.process(MultimodalInput(vision=probe.vision, metadata=probe.metadata))
    audio_output = engine.process(MultimodalInput(audio=probe.audio, metadata=probe.metadata))

    similarity = cosine(vision_output.concept_direction, audio_output.concept_direction)

    assert similarity > 0.6
    assert audio_output.per_modality["vision"].concept_direction is not None


def test_temporal_signal_can_win_workspace_competition() -> None:
    torch.manual_seed(0)
    engine = MultimodalFusionEngine(make_fusion_config())
    for _ in range(8):
        engine.learn(MultimodalInput(temporal_context=[1, 2, 3], metadata={"label": "sequence"}))

    output = engine.process(MultimodalInput(temporal_context=[1, 2, 3], metadata={"label": "sequence"}))

    assert output.winner_modality == "temporal"
    assert output.per_modality["temporal"].concept_direction is not None
    assert output.per_modality["temporal"].top_confidence > 0.0


def test_multimodal_trainer_reports_binding_metrics() -> None:
    torch.manual_seed(0)
    trainer = MultimodalTrainer(make_train_config())
    stream = SyntheticMultimodalStream(
        24,
        num_classes=4,
        image_size=16,
        sample_rate=4000,
        duration_ms=240,
        shuffle=True,
        seed=5,
    )

    metrics = trainer.train_online(stream)

    assert metrics["num_samples"] == 24
    assert metrics["num_passes"] == 2
    assert metrics["num_associations"] > 0
    assert metrics["label_consistency"] > 0.5
    assert metrics["winner_histogram"]
