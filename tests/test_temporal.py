from __future__ import annotations

import torch

from bioarn.config import TemporalConfig, TemporalTrainConfig
from bioarn.data.video import SyntheticVideoStream
from bioarn.temporal import TemporalContextBuffer, TemporalSequenceLayer
from bioarn.training.temporal_training import TemporalTrainer


def _one_hot(index: int, dim: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=dim).float()


def test_temporal_context_buffer_tracks_recent_pattern() -> None:
    buffer = TemporalContextBuffer(window_size=3, concept_dim=4)
    buffer.push(_one_hot(0, 4), [0])
    buffer.push(_one_hot(1, 4), [1])
    buffer.push(_one_hot(1, 4), [1])

    context = buffer.get_context()

    assert context.shape == (4,)
    assert context[1].item() > context[0].item()
    assert buffer.get_temporal_pattern() == [[0], [1], [1]]


def test_temporal_sequence_layer_learns_forward_causality() -> None:
    layer = TemporalSequenceLayer(
        TemporalConfig(
            context_window=4,
            concept_dim=4,
            stdp_tau_plus=8.0,
            stdp_tau_minus=16.0,
            stdp_lr=0.25,
            prediction_threshold=0.2,
        )
    )

    for _ in range(12):
        layer.observe_frame(_one_hot(0, 4), [0])
        layer.observe_frame(_one_hot(1, 4), [1])

    assert layer.temporal_weights[0, 1].item() > 0.0
    layer.reset_state(clear_weights=False)
    output = layer.observe_frame(_one_hot(0, 4), [0])
    assert 1 in output.predicted_indices or int(torch.argmax(output.prediction).item()) == 1


def test_temporal_surprise_spikes_on_causal_violation() -> None:
    layer = TemporalSequenceLayer(
        TemporalConfig(
            context_window=4,
            concept_dim=4,
            stdp_tau_plus=8.0,
            stdp_tau_minus=16.0,
            stdp_lr=0.25,
            prediction_threshold=0.2,
        )
    )
    for _ in range(10):
        layer.observe_frame(_one_hot(0, 4), [0])
        layer.observe_frame(_one_hot(1, 4), [1])

    layer.reset_state(clear_weights=False)
    layer.observe_frame(_one_hot(0, 4), [0])
    expected = layer.observe_frame(_one_hot(1, 4), [1]).surprise

    layer.reset_state(clear_weights=False)
    layer.observe_frame(_one_hot(0, 4), [0])
    violation = layer.observe_frame(_one_hot(2, 4), [2]).surprise

    assert expected < 0.5
    assert violation > expected
    assert violation > 0.5


def test_synthetic_video_stream_emits_temporal_sequences() -> None:
    stream = SyntheticVideoStream(
        num_sequences=6,
        frames_per_sequence=6,
        frame_shape=(16, 16),
        seed=3,
        violation_rate=1.0,
    )

    sequences = list(stream)

    assert len(sequences) == 6
    assert all(len(sequence.frames) == 6 for sequence in sequences)
    assert {sequence.temporal_label for sequence in sequences} >= {"moving_right", "causal_violation"}
    assert sequences[0].frames[0].shape == (16, 16)


def test_temporal_trainer_learns_causal_prediction_and_violation_surprise() -> None:
    config = TemporalTrainConfig(
        frames_per_sequence=8,
        num_sequences=20,
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
    trainer = TemporalTrainer(config)
    for _ in range(24):
        causal = trainer.stream.build_sequence("causal")
        trainer.train_sequence(causal.frames, causal.label, causal.temporal_label)

    causal_result = trainer.train_sequence(
        trainer.stream.build_sequence("causal").frames,
        label=4,
        temporal_label="causal",
    )
    violation_result = trainer.train_sequence(
        trainer.stream.build_sequence("causal_violation").frames,
        label=4,
        temporal_label="causal_violation",
    )
    violation_index = config.frames_per_sequence // 2
    clean_reference_surprise = causal_result.frame_results[violation_index].surprise
    violation_reference_surprise = violation_result.frame_results[violation_index].surprise

    assert causal_result.prediction_accuracy > 0.6
    assert violation_reference_surprise > clean_reference_surprise
    assert violation_reference_surprise > 0.5
