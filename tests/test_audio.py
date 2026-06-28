from __future__ import annotations

import torch

from bioarn.config import AudioConfig, AudioHierarchyConfig, AudioTrainConfig
from bioarn.data import SyntheticAudioStream
from bioarn.hierarchy import AudioHierarchy
from bioarn.preprocessing import AudioPreprocessor
from bioarn.training import AudioTrainer


def test_audio_preprocessor_outputs_mel_shape() -> None:
    config = AudioConfig(sample_rate=8000, n_mels=24, n_fft=256, hop_length=80, max_duration_ms=400)
    preprocessor = AudioPreprocessor(config)
    waveform = torch.sin(2.0 * torch.pi * 220.0 * torch.linspace(0.0, 0.4, int(0.4 * 8000)))

    mel = preprocessor.waveform_to_mel(waveform)

    assert mel.shape == (24, preprocessor.max_frames)
    assert torch.isfinite(mel).all()


def test_audio_preprocessor_flatten_is_fixed_size() -> None:
    config = AudioConfig(sample_rate=8000, n_mels=16, n_fft=256, hop_length=80, max_duration_ms=300)
    preprocessor = AudioPreprocessor(config)
    waveform = torch.randn(int(0.18 * config.sample_rate))

    flattened = preprocessor.to_flat_input(preprocessor.waveform_to_mel(waveform))

    assert flattened.shape == (config.n_mels * preprocessor.max_frames,)


def test_audio_hierarchy_projects_to_belt_dim() -> None:
    preprocessor = AudioPreprocessor(AudioConfig(sample_rate=8000, n_mels=24, n_fft=256, hop_length=80, max_duration_ms=400))
    hierarchy = AudioHierarchy(AudioHierarchyConfig(n_mels=24, a1_channels=12, a2_channels=20, belt_dim=48, temporal_kernel=5))
    waveform = torch.sin(2.0 * torch.pi * 330.0 * torch.linspace(0.0, 0.4, int(0.4 * 8000)))
    mel = preprocessor.waveform_to_mel(waveform)

    concept = hierarchy(mel)

    assert concept.shape == (48,)
    assert torch.isfinite(concept).all()
    assert 0.9 <= float(concept.norm().item()) <= 1.1


def test_synthetic_audio_stream_yields_audio_samples() -> None:
    stream = SyntheticAudioStream(12, sample_rate=8000, duration_ms=250, shuffle=False, seed=3)
    sample = next(stream.stream())

    assert sample.modality == "audio"
    assert sample.data.shape == (2000,)
    assert 0 <= int(sample.label) <= 9
    assert sample.metadata["dataset"] == "synthetic-audio"


def test_synthetic_audio_classes_are_distinguishable() -> None:
    stream = SyntheticAudioStream(10, sample_rate=8000, duration_ms=300, shuffle=False, seed=4)
    samples = list(stream.stream())

    difference = torch.mean(torch.abs(samples[0].data - samples[4].data)).item()

    assert difference > 0.05


def test_audio_trainer_reaches_above_chance_accuracy() -> None:
    config = AudioTrainConfig(
        audio=AudioConfig(sample_rate=8000, n_mels=24, n_fft=256, hop_length=80, max_duration_ms=400),
        hierarchy=AudioHierarchyConfig(n_mels=24, a1_channels=12, a2_channels=24, belt_dim=64, temporal_kernel=5),
        max_pool_size=64,
        margin_threshold=0.97,
        learning_rate=0.01,
        num_train_samples=180,
        num_test_samples=60,
        num_passes=2,
        seed=5,
    )
    trainer = AudioTrainer(config)
    train_stream = SyntheticAudioStream(
        config.num_train_samples,
        sample_rate=config.audio.sample_rate,
        duration_ms=config.audio.max_duration_ms,
        shuffle=True,
        seed=config.seed,
    )
    test_stream = SyntheticAudioStream(
        config.num_test_samples,
        sample_rate=config.audio.sample_rate,
        duration_ms=config.audio.max_duration_ms,
        shuffle=False,
        seed=config.seed + 1,
    )

    train_metrics = trainer.train_online(train_stream)
    eval_metrics = trainer.evaluate(test_stream)

    assert train_metrics["committed_cccs"] >= 10
    assert float(eval_metrics["accuracy"]) > 0.5
    assert float(eval_metrics["coverage"]) > 0.5
