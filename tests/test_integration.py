import torch

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, MarginGateConfig, SDMConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.system import BioARNCore, ContinualLearningEvaluator
from bioarn.workspace.gnw import StreamOfConsciousness


def make_config(max_pool_size: int = 20) -> BioARNConfig:
    return BioARNConfig(
        ccc=CCCConfig(
            input_dim=4,
            concept_dim=4,
            num_f1_features=4,
            f1_top_k=1,
            fast_lr=1.0,
            slow_lr=0.2,
            feedback_lr=0.3,
            max_pool_size=max_pool_size,
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


def configure_identity_core(core: BioARNCore) -> None:
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

        hard_locations = torch.tensor(
            [[float((value >> shift) & 1) for shift in range(3, -1, -1)] for value in range(16)],
            dtype=torch.float32,
        )
        core.fabric.sdm.hard_locations.copy_(hard_locations)
        core.fabric.sdm.data_matrix.zero_()
        core.fabric.sdm.activation_counts.zero_()


def make_core(max_pool_size: int = 20) -> BioARNCore:
    core = BioARNCore(make_config(max_pool_size=max_pool_size))
    configure_identity_core(core)
    return core


def concept(index: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=4).float()


def batch_for(index: int, repeats: int = 3) -> torch.Tensor:
    return concept(index).repeat(repeats, 1)


def test_core_initialization() -> None:
    core = make_core()

    assert core.ccc_pool is not None
    assert core.fabric is not None
    assert core.gnw is not None
    assert isinstance(core.stream, StreamOfConsciousness)
    assert core.timestep == 0


def test_perceive_single_input() -> None:
    core = make_core()

    perception = core.perceive(concept(0))

    assert perception.timestep == 0
    assert perception.num_fired == 1
    assert perception.is_novel is True
    assert perception.pool_output.recruited is True
    assert perception.broadcast.num_occupied >= 1


def test_perceive_triggers_associations() -> None:
    core = make_core()

    core.perceive(concept(0))
    perception = core.perceive(concept(1))

    assert perception.associations.indices
    assert core.fabric.get_stats()["num_associations"] >= 1


def test_recognize_after_learning() -> None:
    core = make_core()
    core.forward(concept(0), learn=True)

    recognition = core.recognize(concept(0))

    assert recognition.abstained is False
    assert recognition.confidence > 0.0
    assert cosine_similarity(recognition.concept_direction, concept(0)).item() > 0.99


def test_recognize_abstains_on_novel() -> None:
    core = make_core()

    recognition = core.recognize(concept(3))

    assert recognition.abstained is True
    assert recognition.num_hypotheses == 0
    assert core.ccc_pool.get_pool_stats()["num_committed"] == 0


def test_thinking_produces_chain() -> None:
    core = make_core()
    core.perceive(concept(0))
    core.perceive(concept(1))
    core.perceive(concept(2))

    thoughts = core.think(num_steps=3)

    assert len(thoughts) == 3
    assert any(thought.broadcast.num_occupied > 0 for thought in thoughts)
    assert thoughts[-1].thought_chain_length >= 1


def test_forward_with_learning() -> None:
    core = make_core()

    output = core.forward(concept(0), learn=True)

    assert output.learned is True
    assert output.perception.is_novel is True
    assert output.thought.broadcast.num_occupied >= 1
    assert output.system_stats["concepts_learned"] == 1


def test_continual_learning_no_forgetting() -> None:
    evaluator = ContinualLearningEvaluator(make_core())

    result = evaluator.run_sequential_test(
        [
            [batch_for(0)],
            [batch_for(1)],
            [batch_for(2)],
        ]
    )

    assert result.passed is True
    assert result.mean_forgetting <= 0.05
    assert result.stage_accuracies[-1][0] >= 0.95
    assert result.stage_accuracies[-1][1] >= 0.95
    assert result.stage_accuracies[-1][2] >= 0.95


def test_system_stats_populated() -> None:
    core = make_core()
    core.forward(concept(0), learn=True)

    stats = core.get_system_stats()

    assert {"pool", "fabric", "gnw", "timesteps", "concepts_learned", "sparsity"} <= set(stats)
    assert {"num_committed", "fire_rate"} <= set(stats["pool"])
    assert {"num_associations", "temporal_chains_count"} <= set(stats["fabric"])
    assert {"occupancy", "turnover_rate"} <= set(stats["gnw"])


def test_sparsity_constraint() -> None:
    core = make_core(max_pool_size=20)

    perception = core.perceive(concept(0))
    active_fraction = perception.num_fired / core.config.ccc.max_pool_size

    assert active_fraction < 0.1


def test_gnw_receives_winners() -> None:
    core = make_core()

    perception = core.perceive(concept(0))

    assert core.gnw.slots
    assert perception.pool_output.recruited_index in perception.broadcast.indices


def test_novel_input_recruits_ccc() -> None:
    core = make_core()

    perception = core.perceive(concept(0))

    assert perception.is_novel is True
    assert perception.pool_output.recruited_index == 0
    assert core.ccc_pool.get_pool_stats()["num_committed"] == 1
