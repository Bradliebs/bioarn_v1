import torch

from bioarn.config import ConvCCCConfig, MarginGateConfig
from bioarn.core.conv_ccc import ConvCCCPool, ConvF1Layer
from bioarn.training import SyntheticCIFAR10Stream, VisionTrainConfig, VisionTrainer


def make_config(*, max_pool_size: int = 4, lock_threshold: float = 0.8) -> ConvCCCConfig:
    return ConvCCCConfig(
        in_channels=3,
        spatial_size=32,
        num_conv_features=8,
        num_conv_layers=3,
        conv_hidden_channels=(12, 16),
        spatial_grid=4,
        f1_top_k=4,
        fast_lr=0.5,
        slow_lr=0.2,
        feedback_lr=0.2,
        conv_hebbian_lr=0.01,
        conv_competitive_k=4,
        spatial_top_k=4,
        max_pool_size=max_pool_size,
        max_growth_factor=2.0,
        consolidation_strength=0.0,
        lock_threshold=lock_threshold,
    )


def make_margin_config(
    theta_margin: float = 0.2,
    theta_resonance: float = 0.8,
) -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=theta_margin,
        theta_margin_lr=0.01,
        theta_resonance=theta_resonance,
    )


def make_image(seed: int) -> torch.Tensor:
    stream = SyntheticCIFAR10Stream(1, flatten=False, shuffle=False, seed=seed)
    return next(stream.stream()).data


def test_conv_f1_layer_sparse_shape() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
    )
    with torch.no_grad():
        layer.conv1.weight.fill_(0.05)
        layer.conv1.bias.zero_()
        layer.conv2.weight.fill_(0.05)
        layer.conv2.bias.zero_()
        layer.conv3.weight.fill_(0.05)
        layer.conv3.bias.zero_()

    output = layer(torch.ones(3, 32, 32))

    assert output.shape == (layer.output_dim,)
    assert torch.count_nonzero(output).item() == 4


def test_conv_f1_layer_hebbian_updates_all_layers() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
        hebbian_lr=0.01,
        competitive_k=4,
        spatial_top_k=4,
    )
    before_conv1 = layer.conv1.weight.clone()
    before_conv2 = layer.conv2.weight.clone()
    before_conv3 = layer.conv3.weight.clone()

    layer.hebbian_update(torch.ones(3, 32, 32), learning_signal=torch.tensor([1.0]))

    assert not torch.allclose(layer.conv1.weight, before_conv1)
    assert not torch.allclose(layer.conv2.weight, before_conv2)
    assert not torch.allclose(layer.conv3.weight, before_conv3)
    for weight in (layer.conv1.weight, layer.conv2.weight, layer.conv3.weight):
        flat = weight.view(weight.shape[0], -1)
        assert torch.allclose(flat.norm(dim=1), torch.ones(weight.shape[0]), atol=1e-4, rtol=1e-4)


def test_conv_ccc_pool_recruits_from_images() -> None:
    pool = ConvCCCPool(make_config(max_pool_size=3), make_margin_config())

    output = pool(make_image(1), timestep=1)

    assert output.recruited is True
    assert output.recruited_index == 0
    assert pool.cccs[0].is_committed.item() is True
    assert 0 in output.fired_indices


def test_conv_ccc_pool_fires_for_familiar_patterns() -> None:
    pool = ConvCCCPool(make_config(max_pool_size=3), make_margin_config())
    image = make_image(2)

    first = pool(image, timestep=1)
    second = pool(image, timestep=2)

    assert first.recruited is True
    assert second.recruited is False
    assert 0 in second.fired_indices
    assert float(second.outputs[0].confidence.item()) > 0.5
    assert second.outputs[0].resonance is not None
    assert bool(second.outputs[0].resonance.resonated.item()) is True


def test_conv_ccc_locking_prevents_updates() -> None:
    pool = ConvCCCPool(make_config(lock_threshold=0.5), make_margin_config())
    image = make_image(3)

    pool(image, timestep=1)
    pool.update_importance([0], confidences=[0.95])
    ccc = pool.cccs[0]
    before_direction = ccc.concept_direction.clone()
    before_conv1 = pool.shared_f1.conv1.weight.clone()
    before_conv2 = pool.shared_f1.conv2.weight.clone()
    before_conv3 = pool.shared_f1.conv3.weight.clone()

    pool(image, timestep=2)

    assert ccc.is_locked.item() is True
    assert torch.allclose(ccc.concept_direction, before_direction)
    assert torch.allclose(pool.shared_f1.conv1.weight, before_conv1)
    assert torch.allclose(pool.shared_f1.conv2.weight, before_conv2)
    assert torch.allclose(pool.shared_f1.conv3.weight, before_conv3)


def test_conv_ccc_handles_cifar10_shapes() -> None:
    pool = ConvCCCPool(make_config(max_pool_size=2), make_margin_config())
    batch = torch.stack([make_image(4), make_image(5)], dim=0)

    output = pool.preview(batch)
    expected_dim = make_config().concept_dim

    assert len(output.outputs) == 2
    assert output.outputs[0].f1_output.shape == (2, expected_dim)
    assert output.outputs[0].f2_activation.shape == (2, expected_dim)


def test_vision_trainer_uses_conv_ccc() -> None:
    trainer = VisionTrainer(
        VisionTrainConfig(
            input_dim=3072,
            concept_dim=256,
            max_pool_size=6,
            use_batched=False,
            num_train_samples=8,
            num_test_samples=4,
            use_conv_ccc=True,
        )
    )

    result = trainer.train_online(
        SyntheticCIFAR10Stream(8, flatten=False, shuffle=False, seed=10),
        num_samples=8,
    )

    assert trainer.system.ccc_pool.__class__.__name__ == "ConvCCCPool"
    assert result["processed_samples"] == 8
    assert result["committed_cccs"] > 0
