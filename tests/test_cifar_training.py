from __future__ import annotations

import torch

from bioarn.config import CCCConfig, GNWConfig, MarginGateConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.loop import SensorimotorLoop
from bioarn.scaling import BatchedCCCPool
from bioarn.training import SyntheticCIFAR10Stream, VisionTrainConfig, VisionTrainer


def make_config(
    *,
    max_pool_size: int = 128,
    num_train_samples: int = 300,
    num_test_samples: int = 120,
    use_batched: bool = True,
    margin_threshold: float = 0.55,
    workspace: GNWConfig | None = None,
) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=max_pool_size,
        margin_threshold=margin_threshold,
        use_batched=use_batched,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        workspace=workspace,
    )


def make_stream(
    num_samples: int,
    *,
    seed: int = 0,
    class_labels: list[int] | None = None,
    shuffle: bool = True,
) -> SyntheticCIFAR10Stream:
    return SyntheticCIFAR10Stream(
        num_samples,
        flatten=True,
        shuffle=shuffle,
        seed=seed,
        class_labels=class_labels,
    )


def make_margin_config(theta_margin: float = 0.8, theta_resonance: float = 0.95) -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=theta_margin,
        theta_margin_lr=0.01,
        theta_resonance=theta_resonance,
    )


def make_ccc_config(max_pool_size: int = 4) -> CCCConfig:
    return CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=1,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.3,
        max_pool_size=max_pool_size,
    )


def configure_identity_original_pool(pool: CCCPool) -> None:
    with torch.no_grad():
        for ccc in pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(4))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(4)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)


def assert_pool_outputs_match(left, right) -> None:
    assert left.recruited == right.recruited
    assert left.recruited_index == right.recruited_index
    assert left.fired_indices == right.fired_indices
    assert left.abstained_indices == right.abstained_indices
    assert torch.allclose(left.winner_confidences, right.winner_confidences, atol=1e-5)
    for left_output, right_output in zip(left.outputs, right.outputs, strict=False):
        assert left_output.fired == right_output.fired
        assert left_output.abstained == right_output.abstained
        assert torch.allclose(left_output.confidence, right_output.confidence, atol=1e-5)
        assert torch.allclose(left_output.f1_output, right_output.f1_output, atol=1e-5)
        assert torch.allclose(left_output.f2_activation, right_output.f2_activation, atol=1e-5)


def test_vision_trainer_init() -> None:
    trainer = VisionTrainer(make_config())

    assert trainer.config.input_dim == 3072
    assert trainer.config.concept_dim == 256
    assert trainer._pool_stats()["total_concepts"] == 128


def test_train_online_runs() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=100))
    result = trainer.train_online(make_stream(100, seed=1), num_samples=100)

    assert result["processed_samples"] == 100
    assert result["committed_cccs"] > 0


def test_accuracy_above_chance() -> None:
    trainer = VisionTrainer(make_config(max_pool_size=196, num_train_samples=1000, num_test_samples=300))
    trainer.train_online(make_stream(1000, seed=2), num_samples=1000)
    metrics = trainer.evaluate(make_stream(300, seed=3, shuffle=False), num_samples=300)

    assert metrics["accuracy"] > 0.15


def test_ood_abstention() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=240))
    trainer.train_online(make_stream(240, seed=4), num_samples=240)
    noise = torch.rand(160, 3072, generator=torch.Generator().manual_seed(14))
    metrics = trainer.ood_detection_test(noise)

    assert metrics["abstention_rate"] > 0.60


def test_continual_no_forget() -> None:
    trainer = VisionTrainer(make_config(max_pool_size=196, num_train_samples=500, num_test_samples=200))
    result = trainer.continual_learning_test(
        make_stream(2000, seed=5),
        class_order=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    )

    assert result["before_accuracy"] >= 0.15
    assert result["after_accuracy"] >= 0.15
    assert result["forgetting"] < 0.20


def test_ccc_specialization() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=400))
    trainer.train_online(make_stream(400, seed=6), num_samples=400)
    analysis = trainer.get_ccc_analysis()

    assert analysis["specialized_cccs"] > 0
    assert analysis["mean_purity"] > 0.50


def test_pool_grows_with_training() -> None:
    trainer = VisionTrainer(make_config(max_pool_size=196))
    trainer.train_online(make_stream(60, seed=7, class_labels=[0, 1, 2, 3, 4]), num_samples=60)
    before = trainer.get_ccc_analysis()["committed_cccs"]
    trainer.train_online(make_stream(240, seed=8), num_samples=240)
    after = trainer.get_ccc_analysis()["committed_cccs"]

    assert after > before


def test_sparsity_maintained() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=320, num_test_samples=120))
    trainer.train_online(make_stream(320, seed=9), num_samples=320)
    metrics = trainer.evaluate(make_stream(120, seed=10, shuffle=False), num_samples=120)

    assert metrics["mean_firing_fraction"] < 0.20


def test_evaluation_metrics() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=180, num_test_samples=80))
    trainer.train_online(make_stream(180, seed=11), num_samples=180)
    metrics = trainer.evaluate(make_stream(80, seed=12, shuffle=False), num_samples=80)

    assert {
        "accuracy",
        "covered_accuracy",
        "abstention_rate",
        "coverage",
        "per_class_accuracy",
        "pool_utilization",
        "mean_firing_count",
        "mean_firing_fraction",
        "total_samples",
    }.issubset(metrics.keys())


def test_workspace_training_path_runs() -> None:
    trainer = VisionTrainer(
        make_config(
            num_train_samples=80,
            num_test_samples=40,
            workspace=GNWConfig(
                capacity=5,
                broadcast_gain=2.2,
                fatigue_rate=0.08,
                fatigue_threshold=0.18,
                competition_temp=0.45,
                context_size=48,
            ),
        )
    )
    trainer.train_online(make_stream(80, seed=21), num_samples=80)
    metrics = trainer.evaluate(make_stream(40, seed=22, shuffle=False), num_samples=40)

    assert trainer.system.config.workspace is not None
    assert metrics["accuracy"] >= 0.0
    assert metrics["abstention_rate"] >= 0.0


def test_batched_vs_sequential() -> None:
    config = make_ccc_config(max_pool_size=4)
    margin = make_margin_config()
    original = CCCPool(config, margin)
    optimized = BatchedCCCPool(config, margin).load_from_pool(original)

    configure_identity_original_pool(original)
    optimized.load_from_pool(original)
    sample = torch.tensor([1.0, 0.8, 0.2, 0.0])

    original_output = original(sample, timestep=1)
    optimized_output = optimized(sample, timestep=1)

    assert_pool_outputs_match(original_output, optimized_output)


def test_no_backprop_cifar() -> None:
    trainer = VisionTrainer(make_config(num_train_samples=120))
    trainer.train_online(make_stream(120, seed=13), num_samples=120)

    assert all(parameter.grad is None for parameter in trainer.system.parameters())


def test_cifar_config_correct_dims() -> None:
    config = VisionTrainConfig()

    assert config.input_dim == 3072
    assert config.concept_dim == 256
    assert SensorimotorLoop._infer_visual_shape(config.input_dim) == (3, 32, 32)
