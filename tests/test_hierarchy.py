"""Tests for the predictive hierarchy and CCC connector."""

from __future__ import annotations

import torch

from bioarn.config import CCCConfig, MarginGateConfig, PredictiveConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.predictive.hierarchy import HierarchyConnector, PredictiveHierarchy


def make_predictive_config(**overrides: float) -> PredictiveConfig:
    config = PredictiveConfig(gamma=0.2, eta=0.01, precision_init=1.0, error_threshold=0.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_ccc_pool(input_dim: int = 4, concept_dim: int = 4, max_pool_size: int = 2) -> CCCPool:
    ccc_config = CCCConfig(
        input_dim=input_dim,
        concept_dim=concept_dim,
        num_f1_features=input_dim,
        f1_top_k=input_dim,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.2,
        max_pool_size=max_pool_size,
    )
    margin_config = MarginGateConfig(theta_margin=0.5, theta_margin_lr=0.01, theta_resonance=0.9)
    pool = CCCPool(ccc_config, margin_config)
    with torch.no_grad():
        for ccc in pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(input_dim))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(concept_dim, input_dim)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
    return pool


def make_identity_hierarchy() -> PredictiveHierarchy:
    hierarchy = PredictiveHierarchy(layer_dims=[4, 4, 4, 4], config=make_predictive_config())
    with torch.no_grad():
        for layer in hierarchy.layers:
            layer.W.copy_(torch.eye(4))
            layer.precision.fill_(1.0)
            layer.state.zero_()
    hierarchy.reset()
    return hierarchy


def test_hierarchy_perceive_shape() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.tensor([1.0, 0.5, 0.0, 0.25]), num_iterations=6)

    assert len(output.states) == 4
    assert len(output.errors) == 4
    assert [state.shape for state in output.states] == [(4,), (4,), (4,), (4,)]
    assert [error.shape for error in output.errors] == [(4,), (4,), (4,), (4,)]


def test_hierarchy_free_energy_decreases() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.tensor([1.0, 0.8, 0.2, 0.4]), num_iterations=8)

    assert output.free_energy_trace[-1] <= output.free_energy_trace[0]


def test_hierarchy_settle_states_is_pure_inference_when_disabled_learning() -> None:
    hierarchy = make_identity_hierarchy()
    original_weights = [layer.W.detach().clone() for layer in hierarchy.layers]

    output = hierarchy.settle_states(
        [
            torch.tensor([0.9, 0.2, 0.1, 0.0]),
            torch.tensor([0.8, 0.1, 0.0, 0.0]),
            torch.tensor([0.7, 0.1, 0.0, 0.0]),
            torch.tensor([0.6, 0.0, 0.0, 0.0]),
        ],
        num_iterations=4,
        learn=False,
    )

    assert len(output.states) == 4
    assert output.free_energy_trace[-1] <= output.free_energy_trace[0]
    for before, layer in zip(original_weights, hierarchy.layers, strict=False):
        assert torch.allclose(before, layer.W)


def test_hierarchy_convergence() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.zeros(4), num_iterations=10)

    assert output.converged is True
    assert output.iterations_used < 10


def test_hierarchy_generation() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.generate(torch.tensor([0.8, 0.3, 0.1, 0.0]))

    assert output.generated_sensory.shape == (4,)
    assert len(output.level_predictions) == 3
    assert torch.all(output.generated_sensory >= 0.0)


def test_hierarchy_predict_and_compare() -> None:
    hierarchy = make_identity_hierarchy()
    familiar = torch.tensor([0.9, 0.1, 0.4, 0.2])
    novel = torch.tensor([0.0, 1.0, 1.0, 0.0])

    hierarchy.perceive(familiar, num_iterations=12)
    familiar_quality = hierarchy.predict_and_compare(familiar)
    novel_quality = hierarchy.predict_and_compare(novel)

    assert familiar_quality.surprise_score < novel_quality.surprise_score
    assert familiar_quality.novel is False
    assert novel_quality.novel is True


def test_active_inference_signal() -> None:
    hierarchy = make_identity_hierarchy()

    action = hierarchy.active_inference_step(torch.zeros(4), torch.ones(4))

    assert torch.count_nonzero(action.direction).item() > 0
    assert action.urgency > 0.0
    assert action.expected_reduction > 0.0


def test_connector_bottom_up() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    concept_level = connector.bottom_up(torch.tensor([1.0, 0.0, 0.5, 0.25]))

    assert concept_level.shape == (4,)


def test_connector_top_down() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    sensory = connector.top_down(torch.tensor([0.7, 0.2, 0.1, 0.0]))

    assert sensory.shape == (4,)
    assert torch.all(sensory >= 0.0)


def test_resonance_loop_converges() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())
    sensory = torch.tensor([0.8, 0.3, 0.0, 0.4])

    output = connector.resonance_loop(sensory, sensory, max_iters=6)

    assert output.resonated is True
    assert output.final_error < 0.1


def test_resonance_triggers_on_match() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())
    sensory = torch.tensor([0.6, 0.2, 0.1, 0.0])

    output = connector.resonance_loop(sensory, sensory, max_iters=4)

    assert output.resonated is True
    assert output.iterations >= 1


def test_resonance_fails_on_mismatch() -> None:
    hierarchy = make_identity_hierarchy()
    with torch.no_grad():
        for layer in hierarchy.layers:
            layer.W.zero_()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    output = connector.resonance_loop(
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        max_iters=4,
    )

    assert output.resonated is False
    assert output.final_error > 0.1


def test_generation_different_concepts() -> None:
    hierarchy = make_identity_hierarchy()

    first = hierarchy.generate(torch.tensor([0.9, 0.0, 0.0, 0.0])).generated_sensory
    second = hierarchy.generate(torch.tensor([0.0, 0.9, 0.0, 0.0])).generated_sensory

    assert not torch.allclose(first, second)


from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import VisionTrainConfig, VisionTrainer


def make_visual_hierarchy_config() -> HierarchyConfig:
    return HierarchyConfig(
        pool_sizes=[20, 28, 36, 20],
        concept_dims=[12, 20, 28, 14],
        thresholds=[0.2, 0.28, 0.34, 0.4],
        learning_rates=[0.05, 0.04, 0.03, 0.02],
    )


def make_unsupervised_visual_config() -> HierarchyConfig:
    return HierarchyConfig(
        pool_sizes=[40, 24, 24, 12],
        concept_dims=[12, 18, 24, 12],
        thresholds=[0.55, 0.35, 0.4, 0.45],
        learning_rates=[0.05, 0.04, 0.03, 0.02],
    )


def make_structured_visual_image(label: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(3, 32, 32, generator=generator) * 0.18
    row = (label % 5) * 5
    col = (label // 5) * 14
    image[label % 3, row : row + 6, :] += 0.6
    image[(label + 1) % 3, :, col : col + 10] += 0.4
    image[(label + 2) % 3, 8:24, 8:24] += 0.05 * label
    return image


def make_visual_train_test_sets(
    *,
    train_samples: int = 240,
    test_samples: int = 120,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    train = [
        (make_structured_visual_image(index % 10, index), index % 10)
        for index in range(train_samples)
    ]
    test = [
        (make_structured_visual_image(index % 10, 1000 + index), index % 10)
        for index in range(test_samples)
    ]
    return train, test


def train_visual_hierarchy(
    config: HierarchyConfig,
    samples: list[tuple[torch.Tensor, int]],
) -> VisualHierarchy:
    hierarchy = VisualHierarchy(config)
    for image, label in samples:
        hierarchy.learn(image, label=label)
    return hierarchy


def test_hierarchy_init() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())

    assert len(hierarchy.layers) == 4
    assert hierarchy.layers[0].name == "V1"
    assert hierarchy.layers[3].name == "IT"
    assert hierarchy.binding is not None


def test_hierarchy_process_shape() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())

    output = hierarchy.process(make_structured_visual_image(2, 7))

    assert output.layer_activations[0].shape == (16, 12)
    assert output.layer_activations[1].shape == (4, 20)
    assert output.layer_activations[2].shape == (1, 28)
    assert output.layer_activations[3].shape == (1, 14)


def test_visual_hierarchy_predictive_refinement_opt_in() -> None:
    hierarchy = VisualHierarchy(
        HierarchyConfig(
            pool_sizes=[20, 28, 36, 20],
            concept_dims=[12, 20, 28, 14],
            thresholds=[0.2, 0.28, 0.34, 0.4],
            learning_rates=[0.05, 0.04, 0.03, 0.02],
            predictive=PredictiveConfig(
                gamma=0.15,
                eta=0.01,
                precision_init=1.0,
                error_threshold=0.0,
                settling_steps=4,
            ),
        )
    )
    for index in range(12):
        hierarchy.learn(make_structured_visual_image(index % 3, index), label=index % 3)

    output = hierarchy.process(make_structured_visual_image(2, 77))

    assert output.predictive_states
    assert output.predictive_errors
    assert output.predictive_free_energy_trace[-1] <= output.predictive_free_energy_trace[0]
    assert output.final_features.shape == (1, 14)


def test_l1_patches_correct() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())

    output = hierarchy.process(make_structured_visual_image(1, 3).reshape(-1))

    assert len(output.patches) == 16
    assert output.patch_grid == (4, 4)
    assert output.layer_inputs[0].shape[1] == hierarchy.config.l1_input_dim


def test_l1_learns_features() -> None:
    hierarchy = VisualHierarchy(make_unsupervised_visual_config())
    for index in range(20):
        generator = torch.Generator().manual_seed(index)
        hierarchy.learn(torch.randn(3, 32, 32, generator=generator))

    assert hierarchy.layers[0].pool.committed_count > 10


def test_l2_uses_l1_output() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())

    output = hierarchy.learn(make_structured_visual_image(0, 11), label=0)
    first_group = output.groupings[0][0]
    expected = torch.cat([output.layer_activations[0][index] for index in first_group], dim=0)

    assert hierarchy.layers[1].last_inputs.shape == (4, 4 * hierarchy.config.concept_dims[0])
    assert torch.allclose(hierarchy.layers[1].last_inputs[0], expected, atol=1e-5)


def test_hierarchy_learns_unsupervised() -> None:
    hierarchy = VisualHierarchy(make_unsupervised_visual_config())
    for index in range(20):
        generator = torch.Generator().manual_seed(100 + index)
        hierarchy.learn(torch.randn(3, 32, 32, generator=generator))

    assert hierarchy.layers[0].pool.committed_count > 0
    assert hierarchy.layers[1].pool.committed_count > 0
    assert hierarchy.layers[2].pool.committed_count > 0
    assert hierarchy.layers[3].pool.committed_count == 0


def test_hierarchy_classifies() -> None:
    train, _ = make_visual_train_test_sets(train_samples=160, test_samples=40)
    hierarchy = train_visual_hierarchy(make_visual_hierarchy_config(), train)

    predicted, confidence = hierarchy.classify(make_structured_visual_image(7, 5007))

    assert predicted == 7
    assert confidence > 0.5


def test_hierarchy_abstains_on_noise() -> None:
    train, _ = make_visual_train_test_sets(train_samples=160, test_samples=40)
    hierarchy = train_visual_hierarchy(make_visual_hierarchy_config(), train)

    predicted, confidence = hierarchy.classify(torch.randn(3, 32, 32))

    assert predicted == -1
    assert confidence == 0.0


def test_hierarchy_accuracy_improves() -> None:
    train, test = make_visual_train_test_sets()
    hierarchy = train_visual_hierarchy(make_visual_hierarchy_config(), train)
    hierarchy_accuracy = sum(
        int(hierarchy.classify(image)[0] == label) for image, label in test
    ) / len(test)

    flat = VisionTrainer(
        VisionTrainConfig(
            input_dim=3072,
            concept_dim=64,
            max_pool_size=10,
            margin_threshold=0.6,
            use_batched=True,
            num_train_samples=len(train),
            num_test_samples=len(test),
        )
    )
    flat_train = [(image.reshape(-1), label) for image, label in train]
    flat_test = [(image.reshape(-1), label) for image, label in test]
    flat.train_online(flat_train, num_samples=len(flat_train))
    flat_metrics = flat.evaluate(flat_test, num_samples=len(flat_test))

    assert hierarchy_accuracy > float(flat_metrics["accuracy"])


def test_feature_binding_strengthens() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())
    image = make_structured_visual_image(3, 303)

    first = hierarchy.learn(image, label=3)
    lower = [first.fired_indices[0][index] for index in first.groupings[0][0]]
    higher = first.fired_indices[1][0]
    before = hierarchy.binding.get_strength(0, lower, higher) if hierarchy.binding else 0.0

    for _ in range(4):
        hierarchy.learn(image, label=3)

    after = hierarchy.binding.get_strength(0, lower, higher) if hierarchy.binding else 0.0

    assert after > before


def test_per_layer_features() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())
    image = make_structured_visual_image(4, 44)

    assert hierarchy.get_layer_features(image, 1).shape == (16, 12)
    assert hierarchy.get_layer_features(image, 2).shape == (4, 20)
    assert hierarchy.get_layer_features(image, 3).shape == (1, 28)
    assert hierarchy.get_layer_features(image, 4).shape == (1, 14)


def test_hierarchy_no_backprop() -> None:
    hierarchy = VisualHierarchy(make_visual_hierarchy_config())
    hierarchy.learn(make_structured_visual_image(5, 55), label=5)

    assert all(
        parameter.grad is None
        for layer in hierarchy.layers
        for parameter in layer.pool.core.parameters()
    )
    assert all(
        not buffer.requires_grad
        for layer in hierarchy.layers
        for buffer in layer.pool.core.buffers()
    )
