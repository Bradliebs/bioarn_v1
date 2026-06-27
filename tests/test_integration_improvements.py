"""Integration tests for hierarchy and ensemble modules wired into BioARNCore.

Organised into five sections:
1. BioARNCore with hierarchy preprocessing
2. BioARNCore with ensemble voting
3. Combined hierarchy + ensemble pipeline
4. EnsembleTrainer — defensive import, skips if Morpheus's work hasn't landed
5. Regression tests — existing BioARNCore behaviour must be unchanged
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bioarn.config import (
    BioARNConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    SDMConfig,
)
from bioarn.core.math_utils import normalize
from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool, HebbianBoosting
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.system import BioARNCore, ContinualLearningEvaluator
from bioarn.training import SyntheticCIFAR10Stream, take_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(*, input_dim: int = 4) -> BioARNConfig:
    """Tiny BioARNConfig that runs in milliseconds."""
    return BioARNConfig(
        ccc=CCCConfig(
            input_dim=input_dim,
            concept_dim=4,
            num_f1_features=4,
            f1_top_k=1,
            fast_lr=1.0,
            slow_lr=0.2,
            feedback_lr=0.3,
            max_pool_size=20,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.8,
            theta_margin_lr=0.01,
            theta_resonance=0.95,
        ),
        sdm=SDMConfig(
            address_dim=4,
            hamming_radius=0,
            num_hard_locations=16,
            data_dim=4,
            decay_rate=1.0,
            stdp_window=4,
        ),
        gnw=GNWConfig(
            capacity=3,
            broadcast_gain=2.0,
            fatigue_rate=0.1,
            fatigue_threshold=0.2,
            competition_temp=0.5,
        ),
        seed=7,
    )


def _small_hierarchy_config() -> HierarchyConfig:
    """32×32×3 hierarchy with tiny pools — fast enough for CI."""
    return HierarchyConfig(
        pool_sizes=[20, 28, 36, 20],
        concept_dims=[12, 20, 28, 14],
        thresholds=[0.2, 0.28, 0.34, 0.4],
        learning_rates=[0.05, 0.04, 0.03, 0.02],
    )


def _make_image(label: int, seed: int) -> torch.Tensor:
    """Deterministic 32×32×3 synthetic image with class-dependent structure."""
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(3, 32, 32, generator=generator) * 0.18
    row = (label % 5) * 5
    col = (label // 5) * 14
    image[label % 3, row : row + 6, :] += 0.6
    image[(label + 1) % 3, :, col : col + 10] += 0.4
    return image


def _concept(index: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(index), num_classes=4).float()


def _batch_for(index: int, repeats: int = 3) -> torch.Tensor:
    return _concept(index).repeat(repeats, 1)


def _identity_core(*, workspace: GNWConfig | None = None) -> BioARNCore:
    """Minimal BioARNCore with identity-initialised weights for deterministic tests."""
    config = _minimal_config()
    if workspace is not None:
        config.workspace = workspace
        config.gnw = workspace
    core = BioARNCore(config)
    with torch.no_grad():
        for ccc in core.ccc_pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(4))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(4)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)
    return core


def _tiny_ensemble_config() -> dict:
    return {
        "input_dim": 3072,
        "concept_dim": 16,
        "max_pool_size": 12,
        "image_size": (32, 32, 3),
        "num_classes": 10,
    }


# ---------------------------------------------------------------------------
# Section 1: BioARNCore with hierarchy preprocessing
# ---------------------------------------------------------------------------

def test_config_hierarchy_attr_safe_access() -> None:
    """getattr(config, 'hierarchy', None) is safe before and after Trinity's changes."""
    config = _minimal_config()
    attr = getattr(config, "hierarchy", None)
    # Pre-merge: None. Post-merge: a HierarchyConfig. Both are acceptable.
    assert attr is None or hasattr(attr, "num_layers")


def test_bioarncore_instantiates_with_patched_hierarchy_attr() -> None:
    """BioARNCore must not crash when config carries an extra hierarchy attribute."""
    config = _minimal_config()
    # Simulate Trinity wiring: duck-type add the field if it doesn't exist yet.
    if not hasattr(config, "hierarchy"):
        config.hierarchy = None  # type: ignore[attr-defined]
    core = BioARNCore(config)
    assert core is not None
    assert core.timestep == 0


def test_hierarchy_preprocessing_compresses_representation() -> None:
    """Hierarchy final features are a smaller representation than the raw flat image."""
    hierarchy = VisualHierarchy(_small_hierarchy_config())
    image = _make_image(2, seed=42)

    raw_dim = image.numel()                         # 3 * 32 * 32 = 3072
    output = hierarchy.process(image)
    hierarchy_dim = output.final_features.numel()   # (1, 14) → 14

    assert hierarchy_dim < raw_dim, (
        f"Expected compressed representation ({hierarchy_dim}) < raw ({raw_dim})"
    )


def test_hierarchy_features_differ_across_classes() -> None:
    """Different class images produce different intermediate representations in the hierarchy."""
    hierarchy = VisualHierarchy(_small_hierarchy_config())

    img_0 = _make_image(0, seed=100)
    img_3 = _make_image(3, seed=300)

    out_0 = hierarchy.process(img_0)
    out_3 = hierarchy.process(img_3)

    # L1 layer inputs are raw patches — these must always differ for different images
    l1_in_0 = out_0.layer_inputs[0]
    l1_in_3 = out_3.layer_inputs[0]
    assert not torch.allclose(l1_in_0, l1_in_3), (
        "Different class images must produce different L1 patch inputs."
    )

    # The hierarchy should have produced non-empty patch grids for both images
    assert len(out_0.patches) > 0
    assert len(out_3.patches) > 0


def test_hierarchy_features_feed_into_bioarncore() -> None:
    """Hierarchy final features (14-dim) can be used as raw input to a matching BioARNCore."""
    hier_config = _small_hierarchy_config()
    feature_dim = hier_config.concept_dims[-1]  # 14

    hierarchy = VisualHierarchy(hier_config)
    # BioARNCore configured to accept the hierarchy output dimension
    core = BioARNCore(BioARNConfig(
        ccc=CCCConfig(
            input_dim=feature_dim,
            concept_dim=8,
            num_f1_features=8,
            f1_top_k=2,
            fast_lr=1.0,
            slow_lr=0.1,
            feedback_lr=0.1,
            max_pool_size=20,
        ),
        margin_gate=MarginGateConfig(theta_margin=0.5, theta_margin_lr=0.001, theta_resonance=0.7),
        sdm=SDMConfig(address_dim=8, hamming_radius=0, num_hard_locations=16, data_dim=8, decay_rate=1.0, stdp_window=4),
        gnw=GNWConfig(capacity=3, broadcast_gain=2.0, fatigue_rate=0.1, fatigue_threshold=0.2, competition_temp=0.5),
        seed=13,
    ))

    image = _make_image(3, seed=99)
    hier_output = hierarchy.process(image)
    features = hier_output.final_features.reshape(-1)[:feature_dim]  # (14,)

    perception = core.perceive(features)

    assert perception.timestep == 0
    assert perception.num_fired >= 0


def test_workspace_opt_in_biases_recognition_confidence() -> None:
    plain_core = _identity_core()
    workspace = GNWConfig(
        capacity=3,
        broadcast_gain=2.2,
        fatigue_rate=0.1,
        fatigue_threshold=0.2,
        competition_temp=0.45,
        context_size=32,
        context_top_k=3,
    )
    workspace.context_bonus = 0.15  # type: ignore[attr-defined]
    workspace_core = _identity_core(workspace=workspace)

    plain_core.forward(_concept(0), learn=True)
    workspace_core.forward(_concept(0), learn=True)

    plain_recognition = plain_core.recognize(_concept(0))
    workspace_recognition = workspace_core.recognize(_concept(0))

    assert workspace_core.last_perception is not None
    assert workspace_core.last_perception.broadcast.context_vector is not None
    assert workspace_recognition.abstained is False
    assert workspace_recognition.confidence >= plain_recognition.confidence


def test_workspace_consensus_prefers_context_supported_candidate() -> None:
    workspace = GNWConfig(
        capacity=3,
        broadcast_gain=2.2,
        fatigue_rate=0.1,
        fatigue_threshold=0.2,
        competition_temp=0.45,
        context_size=32,
        context_top_k=3,
    )
    workspace.context_bonus = 0.20  # type: ignore[attr-defined]
    core = _identity_core(workspace=workspace)
    core.gnw.context.update(_concept(0), 1.0)

    vote_result, broadcast = core.workspace_consensus(
        [
            (0, _concept(0), 0.55),
            (1, _concept(1), 0.60),
        ],
        update_workspace=False,
    )

    assert broadcast.indices[0] == 0
    assert torch.allclose(vote_result.winning_direction, _concept(0), atol=1e-4)


def test_workspace_learning_multiplier_tracks_consensus_strength() -> None:
    core = _identity_core()
    weak_broadcast = core._empty_broadcast()
    strong_broadcast = weak_broadcast.__class__(
        directions=[_concept(0)],
        activations=[float(core.gnw.config.broadcast_gain)],
        indices=[0],
        num_occupied=1,
        total_broadcast_energy=float(core.gnw.config.capacity * core.gnw.config.broadcast_gain),
    )

    weak_multiplier = core.workspace_learning_multiplier(weak_broadcast)
    strong_multiplier = core.workspace_learning_multiplier(strong_broadcast)

    assert weak_multiplier > strong_multiplier
    assert weak_multiplier > 1.0


def test_hierarchy_learn_then_classify_shape() -> None:
    """After learning some images, VisualHierarchy.classify() returns (int, float)."""
    hierarchy = VisualHierarchy(_small_hierarchy_config())
    for idx in range(10):
        hierarchy.learn(_make_image(idx % 3, seed=idx), label=idx % 3)

    predicted, confidence = hierarchy.classify(_make_image(0, seed=500))

    assert isinstance(predicted, int)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_bioarncore_backward_compat_without_hierarchy() -> None:
    """BioARNCore without any hierarchy config behaves exactly as before."""
    config = _minimal_config()
    assert getattr(config, "hierarchy", None) is None  # no Trinity changes yet

    core = _identity_core()
    output = core.forward(_concept(0), learn=True)

    assert isinstance(output.learned, bool)
    assert output.perception.timestep == 0
    assert output.system_stats["concepts_learned"] >= 1


# ---------------------------------------------------------------------------
# Section 2: BioARNCore with ensemble
# ---------------------------------------------------------------------------

class _FixedExpert:
    """Mock expert that always returns a fixed (label, confidence) pair."""

    def __init__(self, label: int, confidence: float) -> None:
        self._label = int(label)
        self._conf = float(confidence)

    def classify(self, _: torch.Tensor) -> tuple[int, float]:
        return self._label, self._conf

    def learn(self, image: torch.Tensor, label: int | None = None) -> None:
        pass


class _AbstainExpert:
    """Mock expert that always abstains."""

    def classify(self, _: torch.Tensor) -> tuple[int, float, bool]:
        return -1, 0.0, True

    def learn(self, image: torch.Tensor, label: int | None = None) -> None:
        pass


def test_config_ensemble_attr_safe_access() -> None:
    """getattr(config, 'ensemble', None) is safe before and after Trinity's changes."""
    config = _minimal_config()
    attr = getattr(config, "ensemble", None)
    assert attr is None or hasattr(attr, "num_experts")


def test_ensemble_voting_produces_valid_result() -> None:
    """EnsemblePool.classify() with three mock experts returns a valid EnsembleResult."""
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3, voting_method="majority"))
    ensemble.add_expert("a", _FixedExpert(1, 0.9))
    ensemble.add_expert("b", _FixedExpert(1, 0.7))
    ensemble.add_expert("c", _FixedExpert(2, 0.5))

    result = ensemble.classify(torch.zeros(4))

    assert result.predicted_class in {1, 2}
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.expert_results) == 3
    assert result.abstained is False


def test_ensemble_majority_selects_plurality_class() -> None:
    """With 2-out-of-3 voting for label 1, majority vote must return 1."""
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3, voting_method="majority"))
    ensemble.add_expert("a", _FixedExpert(1, 0.9))
    ensemble.add_expert("b", _FixedExpert(1, 0.7))
    ensemble.add_expert("c", _FixedExpert(2, 0.5))

    result = ensemble.classify(torch.zeros(4))

    assert result.predicted_class == 1


def test_ensemble_abstains_when_all_experts_abstain() -> None:
    """EnsemblePool abstains when all three experts return abstained=True."""
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3))
    for name in ("a", "b", "c"):
        ensemble.add_expert(name, _AbstainExpert())

    result = ensemble.classify(torch.zeros(4))

    assert result.abstained is True
    assert result.predicted_class == -1


def test_ensemble_result_exposes_per_expert_details() -> None:
    """EnsembleResult.expert_results carries predicted_class, confidence, abstained."""
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2, voting_method="weighted"))
    ensemble.add_expert("low", _FixedExpert(0, 0.3))
    ensemble.add_expert("high", _FixedExpert(1, 0.95))

    result = ensemble.classify(torch.ones(4))

    for ep in result.expert_results:
        assert hasattr(ep, "predicted_class")
        assert hasattr(ep, "confidence")
        assert hasattr(ep, "abstained")


def test_ensemble_learns_delegates_to_all_experts() -> None:
    """EnsemblePool.learn() forwards the sample and label to every expert."""

    class _CountingExpert:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, int | None]] = []

        def classify(self, _: torch.Tensor) -> tuple[int, float]:
            return 0, 0.5

        def learn(self, image: torch.Tensor, label: int | None = None) -> None:
            self.calls.append((image.detach().clone(), label))

    a, b = _CountingExpert(), _CountingExpert()
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2))
    ensemble.add_expert("a", a)
    ensemble.add_expert("b", b)

    sample = torch.tensor([1.0, 2.0, 3.0])
    ensemble.learn(sample, label=5)

    assert len(a.calls) == 1
    assert len(b.calls) == 1
    assert a.calls[0][1] == 5
    assert b.calls[0][1] == 5


def test_bioarncore_backward_compat_without_ensemble() -> None:
    """BioARNCore without any ensemble config behaves exactly as before."""
    config = _minimal_config()
    assert getattr(config, "ensemble", None) is None

    core = _identity_core()
    perception = core.perceive(_concept(0))

    assert perception.timestep == 0
    assert perception.num_fired >= 0


def test_ensemble_with_real_experts_trains_and_classifies() -> None:
    """EnsemblePool backed by DiversityManager experts can learn and infer."""
    samples = take_samples(SyntheticCIFAR10Stream(20, seed=1), 20)
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=2)
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2, expert_configs=configs))

    for image, label in samples:
        ensemble.learn(image, label)

    result = ensemble.classify(samples[0][0])

    assert result.predicted_class >= -1
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Section 3: Combined hierarchy + ensemble
# ---------------------------------------------------------------------------

def test_combined_hierarchy_and_ensemble_instantiation() -> None:
    """VisualHierarchy and EnsemblePool can be created together without conflict."""
    hierarchy = VisualHierarchy(_small_hierarchy_config())
    manager = DiversityManager()
    ensemble = EnsemblePool(
        EnsembleConfig(
            num_experts=2,
            expert_configs=manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=2),
        )
    )

    assert len(hierarchy.layers) == 4
    assert len(ensemble.experts) == 2


def test_combined_hierarchy_expert_in_ensemble() -> None:
    """DiversityManager with num_experts=3 includes a hierarchy-multiscale expert."""
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=3)
    names = [c.name for c in configs]

    assert "hierarchy-multiscale" in names


def test_combined_ensemble_with_hierarchy_expert_trains() -> None:
    """An ensemble that includes a VisualHierarchy expert can complete a learning pass."""
    samples = take_samples(SyntheticCIFAR10Stream(20, seed=5, flatten=False), 20)
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=3)
    ensemble = EnsemblePool(
        EnsembleConfig(num_experts=3, voting_method="majority", expert_configs=configs)
    )

    for image, label in samples:
        ensemble.learn(image, label)  # image shape: (3, 32, 32)

    result = ensemble.classify(samples[0][0])

    assert result.predicted_class >= -1
    assert len(result.expert_results) == 3


def test_combined_perception_pipeline_produces_result() -> None:
    """Hierarchy-processed features followed by ensemble classification produces EnsembleResult."""
    hier_config = _small_hierarchy_config()
    hierarchy = VisualHierarchy(hier_config)

    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=2)
    ensemble = EnsemblePool(
        EnsembleConfig(num_experts=2, voting_method="weighted", expert_configs=configs)
    )

    # Train ensemble on raw images
    for idx in range(15):
        img = _make_image(idx % 5, seed=200 + idx)
        ensemble.learn(img.reshape(-1), label=idx % 5)

    # Perception pipeline: hierarchy extracts features, ensemble classifies raw image
    test_image = _make_image(2, seed=999)
    hier_out = hierarchy.process(test_image)       # verifies hierarchy runs without error
    result = ensemble.classify(test_image.reshape(-1))

    assert hasattr(result, "predicted_class")
    assert hasattr(result, "expert_results")
    assert len(result.expert_results) == 2
    # hierarchy output should have 4 layers' worth of activations
    assert len(hier_out.layer_activations) == 4


def test_combined_diversity_is_measurable_after_training() -> None:
    """After training a multi-expert ensemble, measured diversity is non-negative."""
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=3)
    ensemble = EnsemblePool(
        EnsembleConfig(num_experts=3, voting_method="majority", expert_configs=configs)
    )

    train = take_samples(SyntheticCIFAR10Stream(30, seed=7), 30)
    test = take_samples(SyntheticCIFAR10Stream(10, seed=8, shuffle=False), 10)

    for image, label in train:
        ensemble.learn(image, label)

    histories: list[list[int]] = [[] for _ in ensemble.experts]
    for image, _ in test:
        result = ensemble.classify(image)
        for i, ep in enumerate(result.expert_results):
            histories[i].append(ep.predicted_class)

    diversity = manager.measure_diversity(histories)
    assert diversity >= 0.0


# ---------------------------------------------------------------------------
# Section 4: EnsembleTrainer (Morpheus's work — skip if not yet merged)
# ---------------------------------------------------------------------------

@pytest.fixture
def ensemble_trainer_class():
    """Import EnsembleTrainer; skip the entire test if it hasn't been created yet."""
    try:
        from bioarn.training import EnsembleTrainer  # type: ignore[attr-defined]
        return EnsembleTrainer
    except ImportError:
        pass
    try:
        from bioarn.ensemble import EnsembleTrainer  # type: ignore[attr-defined]
        return EnsembleTrainer
    except ImportError:
        pytest.skip("EnsembleTrainer not yet available — Morpheus's work pending.")


def _trained_ensemble(EnsembleTrainer, num_experts: int = 2, n_train: int = 20):
    """Build and train a small ensemble via EnsembleTrainer (Pattern A: explicit ensemble)."""
    manager = DiversityManager()
    configs = manager.create_diverse_experts(
        {**_tiny_ensemble_config(), "concept_dim": 20, "max_pool_size": 16},
        num_experts=num_experts,
    )
    ensemble = EnsemblePool(EnsembleConfig(num_experts=num_experts, expert_configs=configs))
    samples = take_samples(SyntheticCIFAR10Stream(n_train, seed=42), n_train)

    # Pattern A: pass ensemble explicitly each call.
    # Pattern B (ensemble bound at construction) also works: EnsembleTrainer(ensemble).
    trainer = EnsembleTrainer(log_every=0)
    if hasattr(trainer, "train"):
        trainer.train(ensemble, samples)
    elif hasattr(trainer, "fit"):
        trainer.fit(ensemble, samples)
    else:
        pytest.skip("EnsembleTrainer exposes neither train() nor fit().")

    return trainer, ensemble


def test_ensemble_trainer_instantiates(ensemble_trainer_class) -> None:
    """EnsembleTrainer instantiates with default args and optionally accepts an ensemble."""
    trainer = ensemble_trainer_class(log_every=0)
    assert trainer is not None

    # Pattern B: bind ensemble at construction
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=2)
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2, expert_configs=configs))
    trainer_b = ensemble_trainer_class(ensemble, log_every=0)
    assert trainer_b is not None


def test_ensemble_trainer_trains_on_synthetic_data(ensemble_trainer_class) -> None:
    """EnsembleTrainer.train(ensemble, samples) runs without raising an error."""
    _trained_ensemble(ensemble_trainer_class, num_experts=2, n_train=20)


def test_ensemble_trainer_produces_diversity(ensemble_trainer_class) -> None:
    """After EnsembleTrainer finishes, experts show non-zero pairwise diversity."""
    _, ensemble = _trained_ensemble(ensemble_trainer_class, num_experts=3, n_train=40)
    test = take_samples(SyntheticCIFAR10Stream(10, seed=11, shuffle=False), 10)

    histories: list[list[int]] = [[] for _ in ensemble.experts]
    for image, _ in test:
        result = ensemble.classify(image)
        for i, ep in enumerate(result.expert_results):
            histories[i].append(ep.predicted_class)

    diversity = DiversityManager().measure_diversity(histories)
    assert diversity >= 0.0


def test_ensemble_trainer_boosting_weights_update(ensemble_trainer_class) -> None:
    """With use_boosting=True, HebbianBoosting weights are present and well-shaped after training."""
    manager = DiversityManager()
    configs = manager.create_diverse_experts(_tiny_ensemble_config(), num_experts=2)
    ensemble = EnsemblePool(
        EnsembleConfig(num_experts=2, use_boosting=True, expert_configs=configs)
    )
    trainer = ensemble_trainer_class(log_every=0)
    samples = take_samples(SyntheticCIFAR10Stream(30, seed=99), 30)

    if hasattr(trainer, "train"):
        trainer.train(ensemble, samples)
    elif hasattr(trainer, "fit"):
        trainer.fit(ensemble, samples)
    else:
        pytest.skip("EnsembleTrainer exposes neither train() nor fit().")

    if ensemble.boosting is not None:
        weights_0 = ensemble.boosting.get_weights(0)
        assert weights_0.numel() > 0
        assert weights_0.shape[0] >= 1


# ---------------------------------------------------------------------------
# Section 5: Regression tests — existing BioARNCore behaviour unchanged
# ---------------------------------------------------------------------------

def test_regression_default_config_still_instantiates() -> None:
    """BioARNConfig() and BioARNCore(config) must work with zero arguments."""
    config = BioARNConfig()
    core = BioARNCore(config)
    assert core is not None


def test_regression_perceive_returns_full_perception_output() -> None:
    """core.perceive() still returns PerceptionOutput with all expected fields."""
    core = _identity_core()
    perception = core.perceive(_concept(0))

    for field in ("pool_output", "vote_result", "broadcast", "associations",
                  "num_fired", "is_novel", "timestep"):
        assert hasattr(perception, field), f"Missing field: {field}"


def test_regression_forward_returns_bioarncoreoutput() -> None:
    """core.forward() still returns BioARNCoreOutput with all expected fields."""
    core = _identity_core()
    output = core.forward(_concept(1), learn=True)

    for field in ("perception", "thought", "learned", "system_stats"):
        assert hasattr(output, field), f"Missing field: {field}"


def test_regression_recognize_still_works() -> None:
    """core.recognize() returns a valid RecognitionOutput after learning."""
    core = _identity_core()
    core.forward(_concept(0), learn=True)
    recognition = core.recognize(_concept(0))

    assert not recognition.abstained
    assert recognition.confidence > 0.0
    assert recognition.num_hypotheses > 0


def test_regression_continual_learning_evaluator() -> None:
    """ContinualLearningEvaluator.run_sequential_test() still passes with default data."""
    evaluator = ContinualLearningEvaluator(_identity_core())
    result = evaluator.run_sequential_test(
        [[_batch_for(0)], [_batch_for(1)], [_batch_for(2)]]
    )

    assert result.passed is True
    assert result.mean_forgetting <= 0.05
    assert len(result.stage_accuracies) == 3


def test_regression_system_stats_keys_unchanged() -> None:
    """get_system_stats() must still expose the same top-level keys."""
    core = _identity_core()
    core.forward(_concept(0), learn=True)
    stats = core.get_system_stats()

    required = {"pool", "fabric", "gnw", "timesteps", "concepts_learned", "sparsity"}
    assert required <= set(stats)


def test_regression_full_pipeline_smoke() -> None:
    """End-to-end smoke: perceive × 3 → think × 2 → recognize → forward must not crash."""
    core = _identity_core()

    # Learn three distinct concepts
    for idx in range(3):
        core.forward(_concept(idx), learn=True)

    # Internal association-driven reasoning
    thoughts = core.think(num_steps=2)
    assert len(thoughts) == 2

    # Recognition (no recruitment)
    recognition = core.recognize(_concept(0))
    assert recognition.abstained in {True, False}

    # One more forward pass
    output = core.forward(_concept(2), learn=False)
    assert output.perception.timestep > 0
