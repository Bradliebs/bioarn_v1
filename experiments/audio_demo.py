"""Demo: Bio-ARN synthetic audio classification."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioarn.config import AudioConfig, AudioHierarchyConfig, AudioTrainConfig
from bioarn.data import SyntheticAudioStream
from bioarn.training import AudioTrainer


def main() -> None:
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

    print("=== Bio-ARN Audio Demo ===")
    print(f"train_samples: {config.num_train_samples}")
    print(f"test_samples: {config.num_test_samples}")
    print(f"committed_cccs: {train_metrics['committed_cccs']}")
    print(f"train_accuracy: {float(train_metrics['accuracy']):.3f}")
    print(f"test_accuracy: {float(eval_metrics['accuracy']):.3f}")
    print(f"coverage: {float(eval_metrics['coverage']):.3f}")
    print(f"abstention_rate: {float(eval_metrics['abstention_rate']):.3f}")
    print(f"pool_utilization: {float(eval_metrics['pool_utilization']):.3f}")


if __name__ == "__main__":
    main()
