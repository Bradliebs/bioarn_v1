from __future__ import annotations

import torch

from bioarn.data.base import DataSample
from bioarn.multimodal import (
    AlignmentMetrics,
    ModalityAligner,
    MultimodalConfig,
    MultimodalFusion,
    SpikeCaptioner,
)
from bioarn.training import TextGenConfig, TextGenerationTrainer


def make_config(**overrides) -> MultimodalConfig:
    config = MultimodalConfig(
        vision_dim=28 * 28,
        language_dim=64,
        concept_dim=64,
        cross_modal_strength=0.75,
        temporal_window=3,
        max_description_length=12,
        alignment_threshold=0.55,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_fusion(**overrides) -> MultimodalFusion:
    torch.manual_seed(0)
    return MultimodalFusion(make_config(**overrides))


def make_text_trainer() -> TextGenerationTrainer:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=96,
        context_length=12,
        spike_dim=48,
        num_timesteps=4,
        max_pool_size=48,
        temperature=1.0,
        learning_rate_hebbian=0.02,
        sdm_addresses=128,
        generate_max_tokens=24,
    )
    return TextGenerationTrainer(config)


def pattern(kind: str) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    if kind == "horizontal":
        image[14, 4:24] = 1.0
    elif kind == "vertical":
        image[4:24, 14] = 1.0
    elif kind == "diagonal":
        idx = torch.arange(5, 23)
        image[idx, idx] = 1.0
    elif kind == "anti_diagonal":
        idx = torch.arange(5, 23)
        image[idx, 27 - idx] = 1.0
    elif kind == "box":
        image[6, 6:22] = 1.0
        image[21, 6:22] = 1.0
        image[6:22, 6] = 1.0
        image[6:22, 21] = 1.0
    else:
        raise ValueError(f"Unsupported pattern: {kind}")
    return image


def noisy_variant(base: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = 0.03 * torch.randn(base.shape, generator=generator)
    return (base + noise).clamp_(0.0, 1.0)


def train_basic_pairs(fusion: MultimodalFusion) -> list[tuple[torch.Tensor, str]]:
    pairs = [
        (pattern("horizontal"), "horizontal"),
        (pattern("vertical"), "vertical"),
        (pattern("diagonal"), "diagonal"),
    ]
    for image, label in pairs:
        fusion.learn_cross_modal(image, label, label=label)
    return pairs


def test_fusion_init() -> None:
    fusion = make_fusion()

    assert fusion.config.vision_dim == 784
    assert fusion.visual_encoder.output_dim == 64
    assert fusion.language_encoder.output_dim == 64


def test_bind_visual_to_text() -> None:
    fusion = make_fusion()
    binding = fusion.learn_cross_modal(pattern("horizontal"), "horizontal", label="horizontal")

    fusion.bind_visual_to_text(binding["visual_ccc_id"], binding["text_ccc_id"], strength=2.0)
    result = fusion.cross_modal_retrieval(pattern("horizontal"), "vision", "text")

    assert result
    assert result[0].label == "horizontal"
    assert result[0].strength >= 2.0


def test_learn_cross_modal() -> None:
    fusion = make_fusion()

    binding = fusion.learn_cross_modal(pattern("vertical"), "vertical", label="vertical")

    assert binding["visual_ccc_id"] != binding["text_ccc_id"]
    assert ("vision", "vertical") in fusion.label_to_ccc
    assert ("text", "vertical") in fusion.label_to_ccc
    assert (binding["visual_ccc_id"], binding["text_ccc_id"]) in fusion.fabric.association_strength


def test_describe_image_produces_output() -> None:
    fusion = make_fusion()
    train_basic_pairs(fusion)

    description = fusion.describe_image(noisy_variant(pattern("diagonal"), seed=11))

    assert description
    assert "diagonal" in description


def test_visualize_text_produces_tensor() -> None:
    fusion = make_fusion()
    train_basic_pairs(fusion)

    visual = fusion.visualize_text("vertical")

    assert isinstance(visual, torch.Tensor)
    assert visual.numel() == 28 * 28
    assert float(visual.sum().item()) > 0.0


def test_cross_modal_retrieval() -> None:
    fusion = make_fusion()
    train_basic_pairs(fusion)

    result = fusion.cross_modal_retrieval(noisy_variant(pattern("horizontal"), seed=7), "vision", "text")

    assert result
    assert result[0].label == "horizontal"


def test_alignment_by_label() -> None:
    fusion = make_fusion()
    aligner = ModalityAligner(fusion)
    labels = ["horizontal", "vertical", "diagonal"]

    bindings = aligner.align_by_label(
        [pattern("horizontal"), pattern("vertical"), pattern("diagonal")],
        labels,
        labels,
    )

    assert len(bindings) == 3
    assert fusion.cross_modal_retrieval(pattern("vertical"), "vision", "text")[0].label == "vertical"


def test_alignment_by_cooccurrence() -> None:
    fusion = make_fusion()
    aligner = ModalityAligner(fusion)
    horizontal_tokens = torch.tensor(fusion.tokenizer.encode("horizontal"), dtype=torch.long)
    vertical_tokens = torch.tensor(fusion.tokenizer.encode("vertical"), dtype=torch.long)
    stream = [
        DataSample(data=pattern("horizontal"), label=None, modality="vision"),
        DataSample(
            data=horizontal_tokens,
            label=None,
            modality="text",
            metadata={"text": "horizontal"},
        ),
        DataSample(data=pattern("vertical"), label=None, modality="vision"),
        DataSample(
            data=vertical_tokens,
            label=None,
            modality="text",
            metadata={"text": "vertical"},
        ),
    ]

    bindings = aligner.align_by_co_occurrence(stream)

    assert bindings
    assert fusion.cross_modal_retrieval(pattern("horizontal"), "vision", "text")[0].label == "horizontal"


def test_alignment_metrics() -> None:
    fusion = make_fusion()
    aligner = ModalityAligner(fusion)
    pairs = train_basic_pairs(fusion)

    metrics = aligner.measure_alignment(
        [(noisy_variant(image, seed=index + 1), label, label) for index, (image, label) in enumerate(pairs)]
    )

    assert isinstance(metrics, AlignmentMetrics)
    assert metrics.num_pairs == 3
    assert 0.0 <= metrics.retrieval_accuracy <= 1.0
    assert metrics.mean_reciprocal_rank >= 0.0
    assert metrics.mean_association_strength > 0.0


def test_captioner_produces_text() -> None:
    fusion = make_fusion()
    trainer = make_text_trainer()
    captioner = SpikeCaptioner(fusion, trainer)

    captioner.train_captioning(
        [
            (pattern("horizontal"), "horizontal line"),
            (pattern("vertical"), "vertical line"),
            (pattern("diagonal"), "diagonal line"),
        ],
        num_pairs=3,
    )
    caption = captioner.caption(pattern("horizontal"))

    assert caption
    assert isinstance(caption, str)


def test_bidirectional_binding() -> None:
    fusion = make_fusion()
    fusion.learn_cross_modal(pattern("box"), "box", label="box")

    image_to_text = fusion.cross_modal_retrieval(pattern("box"), "vision", "text")
    text_to_image = fusion.cross_modal_retrieval("box", "text", "vision")

    assert image_to_text and image_to_text[0].label == "box"
    assert text_to_image
    assert torch.count_nonzero(fusion.visualize_text("box")) > 0


def test_no_cross_contamination() -> None:
    fusion = make_fusion()
    fusion.learn_cross_modal(pattern("horizontal"), "horizontal", label="horizontal")

    unrelated_image_results = fusion.cross_modal_retrieval(pattern("anti_diagonal"), "vision", "text")
    unrelated_text_results = fusion.cross_modal_retrieval("completely-unseen-token", "text", "vision")

    assert unrelated_image_results == []
    assert unrelated_text_results == []
