"""Tests for the embodied sensorimotor sensory encoders."""

from __future__ import annotations

import torch

from bioarn.config import SpikingConfig
from bioarn.sensorimotor.language import LanguageEncoder
from bioarn.sensorimotor.vision import VisualEncoder


def make_spiking_config() -> SpikingConfig:
    return SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0)


def make_visual_encoder(output_dim: int = 64) -> VisualEncoder:
    return VisualEncoder(input_shape=(1, 28, 28), output_dim=output_dim, config=make_spiking_config())


def make_language_encoder(output_dim: int = 64) -> LanguageEncoder:
    return LanguageEncoder(vocab_size=32, embedding_dim=16, output_dim=output_dim, config=make_spiking_config())


def active_fraction(features: torch.Tensor) -> float:
    return float((features > 0).float().mean().item())


def square_frame(top: int, left: int, size: int = 6, value: float = 1.0) -> torch.Tensor:
    frame = torch.zeros(1, 1, 28, 28)
    frame[:, :, top : top + size, left : left + size] = value
    return frame


def test_vision_encoder_output_shape() -> None:
    encoder = make_visual_encoder()
    frame = square_frame(8, 8)

    output = encoder(frame)

    assert output.features.shape == (1, 64)


def test_vision_delta_encoding() -> None:
    encoder = make_visual_encoder()
    frame = square_frame(10, 10)

    static_output = encoder(frame, prev_frame=frame.clone())
    changed_output = encoder(frame, prev_frame=torch.zeros_like(frame))

    assert static_output.event_count == 0
    assert changed_output.event_count > 0
    assert torch.count_nonzero(changed_output.raw_events) > 0


def test_vision_sparse_output() -> None:
    encoder = make_visual_encoder()
    frame = square_frame(6, 6, size=8)

    output = encoder(frame, prev_frame=torch.zeros_like(frame))

    assert active_fraction(output.features) < 0.1


def test_vision_predictive_suppression() -> None:
    encoder = make_visual_encoder()
    repeated_frame = square_frame(4, 4)
    novel_frame = square_frame(16, 16)

    encoder(repeated_frame, prev_frame=torch.zeros_like(repeated_frame))
    repeated_output = encoder(repeated_frame)
    novel_output = encoder(novel_frame)

    assert repeated_output.suppressed_fraction > 0.9
    assert novel_output.suppressed_fraction < repeated_output.suppressed_fraction


def test_vision_sequence_encoding() -> None:
    encoder = make_visual_encoder()
    frames = torch.stack(
        [
            square_frame(2, 2).squeeze(0),
            square_frame(8, 8).squeeze(0),
            square_frame(14, 14).squeeze(0),
        ],
        dim=0,
    )

    outputs = encoder.encode_sequence(frames)

    assert len(outputs) == 3
    assert outputs[1].event_count > 0
    assert not torch.equal(outputs[1].features, outputs[2].features)


def test_language_encoder_output_shape() -> None:
    torch.manual_seed(0)
    encoder = make_language_encoder()
    token_ids = torch.tensor([[1, 2, 3]])

    output = encoder(token_ids)

    assert output.features.shape == (1, 64)


def test_language_spike_encoding() -> None:
    encoder = make_language_encoder()

    torch.manual_seed(0)
    first = encoder(torch.tensor([[1]]))
    encoder.reset_state()
    torch.manual_seed(0)
    second = encoder(torch.tensor([[2]]))

    assert not torch.equal(first.spike_train, second.spike_train)


def test_language_sparse_output() -> None:
    torch.manual_seed(1)
    encoder = make_language_encoder()
    token_ids = torch.tensor([[1, 2, 3, 4]])

    output = encoder(token_ids)

    assert active_fraction(output.features) < 0.1


def test_language_temporal_features() -> None:
    torch.manual_seed(2)
    encoder = make_language_encoder()
    token_ids = torch.tensor([[1, 2, 3]])

    output = encoder(token_ids)
    blocks = output.spike_train.view(encoder.steps_per_token, 3, 1, encoder.embedding_dim).permute(1, 0, 2, 3)

    assert not torch.equal(blocks[0], blocks[1])
    assert not torch.equal(blocks[1], blocks[2])


def test_mnist_compatible() -> None:
    encoder = make_visual_encoder(output_dim=32)
    frame = torch.rand(2, 1, 28, 28)

    output = encoder(frame, prev_frame=torch.zeros_like(frame))

    assert output.features.shape == (2, 32)
    assert output.raw_events.shape == (2, 1, 28, 28)
