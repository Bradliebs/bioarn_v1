"""Runnable temporal STDP video-learning demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioarn.config import TemporalConfig, TemporalTrainConfig
from bioarn.training.temporal_training import TemporalTrainer


def main() -> None:
    trainer = TemporalTrainer(
        TemporalTrainConfig(
            frames_per_sequence=8,
            num_sequences=24,
            concept_dim=64,
            max_pool_size=32,
            use_workspace_context=True,
            frame_shape=(8, 8),
            seed=5,
            prediction_top_k=6,
            context_gain=0.25,
            causal_violation_rate=0.4,
            temporal=TemporalConfig(
                context_window=8,
                concept_dim=64,
                stdp_tau_plus=10.0,
                stdp_tau_minus=20.0,
                stdp_lr=0.15,
                prediction_threshold=0.2,
            ),
        )
    )

    training_accuracies: list[float] = []
    for _ in range(24):
        calibration_sequence = trainer.stream.build_sequence("causal")
        training_accuracies.append(
            trainer.train_sequence(
                calibration_sequence.frames,
                calibration_sequence.label,
                calibration_sequence.temporal_label,
            ).prediction_accuracy
        )
    clean_sequence = trainer.stream.build_sequence("causal")
    violated_sequence = trainer.stream.build_sequence("causal_violation")
    clean_result = trainer.train_sequence(
        clean_sequence.frames,
        clean_sequence.label,
        clean_sequence.temporal_label,
    )
    violated_result = trainer.train_sequence(
        violated_sequence.frames,
        violated_sequence.label,
        violated_sequence.temporal_label,
    )
    violation_index = trainer.config.frames_per_sequence // 2
    clean_reference_surprise = clean_result.frame_results[violation_index].surprise
    violation_reference_surprise = violated_result.frame_results[violation_index].surprise

    print("=== Bio-ARN Temporal STDP Demo ===")
    print("causal_training_passes: 24")
    print(f"mean_training_accuracy: {sum(training_accuracies) / len(training_accuracies):.3f}")
    print(f"post_train_clean_causal_accuracy: {clean_result.prediction_accuracy:.3f}")
    print(f"post_train_violation_accuracy: {violated_result.prediction_accuracy:.3f}")
    print()
    print(f"post_train_clean_reference_surprise: {clean_reference_surprise:.3f}")
    print(f"post_train_violation_reference_surprise: {violation_reference_surprise:.3f}")
    print(f"surprise_delta: {violation_reference_surprise - clean_reference_surprise:.3f}")


if __name__ == "__main__":
    main()
