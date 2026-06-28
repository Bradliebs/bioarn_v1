import torch

from bioarn.config import ConvCCCConfig, MarginGateConfig, deep_cifar_config
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
        hebbian_batch_size=1,
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
        hebbian_batch_size=1,
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


def test_conv_f1_layer_batched_hebbian_flushes_at_threshold() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
        hebbian_lr=0.01,
        hebbian_batch_size=4,
        competitive_k=4,
        spatial_top_k=4,
    )
    before_conv1 = layer.conv1.weight.clone()

    for seed in range(3):
        applied = layer.hebbian_update(
            make_image(seed),
            learning_signal=torch.tensor([1.0]),
        )
        assert applied is False
        assert torch.allclose(layer.conv1.weight, before_conv1)

    applied = layer.hebbian_update(
        make_image(3),
        learning_signal=torch.tensor([1.0]),
    )

    assert applied is True
    assert not torch.allclose(layer.conv1.weight, before_conv1)
    for weight in (layer.conv1.weight, layer.conv2.weight, layer.conv3.weight):
        flat = weight.view(weight.shape[0], -1)
        assert torch.allclose(flat.norm(dim=1), torch.ones(weight.shape[0]), atol=1e-4, rtol=1e-4)


def test_conv_f1_layer_manual_flush_applies_pending_updates() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
        hebbian_lr=0.01,
        hebbian_batch_size=8,
        competitive_k=4,
        spatial_top_k=4,
    )
    before_conv1 = layer.conv1.weight.clone()

    applied = layer.hebbian_update(
        make_image(10),
        learning_signal=torch.tensor([1.0]),
    )

    assert applied is False
    assert torch.allclose(layer.conv1.weight, before_conv1)
    assert layer.flush_hebbian_updates() is True
    assert not torch.allclose(layer.conv1.weight, before_conv1)


def test_layerwise_trains_one_at_a_time() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        num_layers=3,
        hidden_channels=(12, 16),
        kernel_sizes=(5, 3, 3),
        hebbian_lr=0.01,
        competitive_k=4,
        spatial_top_k=4,
    )
    before_weights = [conv.weight.clone() for conv in layer.conv_layers]

    results = layer.train_layerwise(
        [make_image(20), make_image(21)],
        samples_per_layer=2,
        passes_per_layer=1,
        lr_schedule=[0.01, 0.007, 0.005],
    )

    assert results["layer_0"]["changed_layers"] == [0]
    assert results["layer_1"]["changed_layers"] == [1]
    assert results["layer_2"]["changed_layers"] == [2]
    assert all(result["updates_applied"] > 0 for result in results.values())
    assert all(layer._is_layer_frozen(index) for index in range(layer.num_layers))
    assert all(
        not torch.allclose(conv.weight, before)
        for conv, before in zip(layer.conv_layers, before_weights, strict=True)
    )


def test_layerwise_freezes_previous() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
        hebbian_lr=0.01,
        hebbian_batch_size=1,
        competitive_k=4,
        spatial_top_k=4,
    )
    before_conv1 = layer.conv1.weight.clone()
    before_conv2 = layer.conv2.weight.clone()
    before_conv3 = layer.conv3.weight.clone()
    layer._freeze_layer(0)

    layer.hebbian_update(make_image(22), learning_signal=torch.tensor([1.0]))

    assert layer._is_layer_frozen(0) is True
    assert torch.allclose(layer.conv1.weight, before_conv1)
    assert not torch.allclose(layer.conv2.weight, before_conv2)
    assert not torch.allclose(layer.conv3.weight, before_conv3)


def test_deeper_config_creates_more_layers() -> None:
    config = deep_cifar_config()
    pool = ConvCCCPool(config, make_margin_config())

    assert config.num_conv_layers == 5
    assert config.layerwise_train.enabled is True
    assert len(pool.shared_f1.conv_layers) == 5
    assert pool.shared_f1.feature_channels == (96, 128, 192, 256, 384)


def test_maxpool_reduces_spatial() -> None:
    layer = ConvF1Layer(
        in_channels=3,
        num_features=16,
        spatial_size=32,
        top_k=8,
        num_layers=5,
        hidden_channels=(8, 10, 12, 14),
        kernel_sizes=(5, 3, 3, 3, 3),
        competitive_k=4,
        spatial_top_k=4,
    )

    _, trace = layer._forward_dense(torch.randn(1, 3, 32, 32))

    assert trace.pre_activations[0].shape[-2:] == (32, 32)
    assert trace.pre_activations[1].shape[-2:] == (32, 32)
    assert trace.pre_activations[2].shape[-2:] == (16, 16)
    assert trace.pre_activations[3].shape[-2:] == (16, 16)
    assert trace.pre_activations[4].shape[-2:] == (8, 8)


def test_softhebb_updates_weights() -> None:
    torch.manual_seed(0)
    layer = ConvF1Layer(
        in_channels=3,
        num_features=8,
        spatial_size=32,
        top_k=4,
        hidden_channels=(12, 16),
        hebbian_lr=0.01,
        hebbian_batch_size=1,
        competitive_k=4,
        spatial_top_k=4,
        softhebb_enabled=True,
    )
    before_weights = [conv.weight.clone() for conv in layer.conv_layers]
    before_theta = [theta.clone() for theta in layer.softhebb_thetas]

    layer.hebbian_update(make_image(11), learning_signal=torch.tensor([1.0]))

    assert any(not torch.allclose(conv.weight, before) for conv, before in zip(layer.conv_layers, before_weights, strict=True))
    assert any(not torch.allclose(theta, before) for theta, before in zip(layer.softhebb_thetas, before_theta, strict=True))
    for theta in layer.softhebb_thetas:
        assert torch.all(theta >= 0.0)


def test_softhebb_bcm_threshold_adapts() -> None:
    torch.manual_seed(1)
    layer = ConvF1Layer(
        in_channels=3,
        num_features=6,
        spatial_size=32,
        top_k=4,
        num_layers=1,
        hidden_channels=(),
        kernel_sizes=(3,),
        hebbian_lr=0.01,
        hebbian_batch_size=2,
        competitive_k=2,
        spatial_top_k=4,
        softhebb_enabled=True,
        softhebb_gamma=3.0,
        softhebb_beta=2.0,
        softhebb_theta_decay=0.8,
    )
    initial_theta = layer.softhebb_thetas[0].clone()

    layer.hebbian_update(make_image(12), learning_signal=torch.tensor([1.0]))
    assert torch.allclose(layer.softhebb_thetas[0], initial_theta)

    layer.hebbian_update(make_image(13), learning_signal=torch.tensor([1.0]))
    first_theta = layer.softhebb_thetas[0].clone()
    layer.hebbian_update(make_image(14), learning_signal=torch.tensor([1.0]))
    layer.hebbian_update(make_image(15), learning_signal=torch.tensor([1.0]))
    second_theta = layer.softhebb_thetas[0].clone()

    assert not torch.allclose(first_theta, initial_theta)
    assert not torch.allclose(second_theta, first_theta)
    assert torch.isfinite(second_theta).all()


def test_softhebb_vs_hard_competitive() -> None:
    torch.manual_seed(2)
    shared_kwargs = dict(
        in_channels=3,
        num_features=6,
        spatial_size=32,
        top_k=4,
        num_layers=1,
        hidden_channels=(),
        kernel_sizes=(3,),
        hebbian_lr=0.01,
        hebbian_batch_size=1,
        competitive_k=1,
        spatial_top_k=4,
    )
    hard_layer = ConvF1Layer(**shared_kwargs)
    soft_layer = ConvF1Layer(**shared_kwargs, softhebb_enabled=True, softhebb_gamma=2.0)
    soft_layer.load_state_dict(hard_layer.state_dict(), strict=False)
    image = make_image(16)

    before_hard = hard_layer.conv1.weight.clone()
    before_soft = soft_layer.conv1.weight.clone()
    hard_layer.hebbian_update(image, learning_signal=torch.tensor([1.0]))
    soft_layer.hebbian_update(image, learning_signal=torch.tensor([1.0]))

    hard_delta = (hard_layer.conv1.weight - before_hard).view(hard_layer.conv1.weight.shape[0], -1).norm(dim=1)
    soft_delta = (soft_layer.conv1.weight - before_soft).view(soft_layer.conv1.weight.shape[0], -1).norm(dim=1)
    hard_active = int((hard_delta > 1e-5).sum().item())
    soft_active = int((soft_delta > 1e-5).sum().item())

    assert soft_active > hard_active


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
