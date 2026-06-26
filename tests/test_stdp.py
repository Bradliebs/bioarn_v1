from __future__ import annotations

import torch

from bioarn.config import CCCConfig, MarginGateConfig, STDPConfig
from bioarn.core.ccc import CCCPool, ConceptCellCluster
from bioarn.core.math_utils import normalize
from bioarn.core.stdp import STDPRule


def make_margin_config() -> MarginGateConfig:
    return MarginGateConfig(theta_margin=0.6, theta_margin_lr=0.01, theta_resonance=0.95)


def make_ccc_config(*, stdp: STDPConfig | None) -> CCCConfig:
    return CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=2,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.3,
        max_pool_size=2,
        stdp=stdp,
    )


def configure_identity_ccc(ccc: ConceptCellCluster) -> None:
    with torch.no_grad():
        ccc.f1_layer.weight.copy_(torch.eye(4))
        ccc.f1_layer.bias.zero_()
        ccc.f2_weights.copy_(normalize(torch.eye(4)))
        ccc.feedback_weights.zero_()
        ccc.concept_direction.zero_()
        ccc.is_committed.zero_()
        ccc.age.zero_()
        ccc.last_fired.fill_(-1)
        if ccc.stdp_rule is not None:
            ccc.stdp_rule.reset_state()


def concept_input() -> torch.Tensor:
    return torch.tensor([1.0, 0.8, 0.2, 0.0])


def shifted_input() -> torch.Tensor:
    return torch.tensor([0.2, 1.0, 0.8, 0.0])


def commit_ccc(ccc: ConceptCellCluster) -> None:
    f1_output = ccc.f1_encode(concept_input())
    ccc.learn_fast(concept_input(), f1_output)


def test_pre_before_post_strengthens() -> None:
    rule = STDPRule(
        STDPConfig(tau_plus=10.0, tau_minus=20.0, A_plus=0.3, A_minus=0.2),
        num_pre=3,
        num_post=2,
    )

    rule.observe_pre_spikes(torch.tensor([1.0, 0.0, 1.0]), timestep=0)
    update = rule.step(
        torch.zeros(3),
        post_spike=True,
        post_activity=torch.tensor([1.0, 0.5]),
        timestep=1,
    )

    assert float(update[0, 0].item()) > 0.0
    assert float(update[2, 0].item()) > 0.0
    assert torch.all(update >= 0.0)


def test_post_before_pre_weakens() -> None:
    rule = STDPRule(
        STDPConfig(tau_plus=10.0, tau_minus=20.0, A_plus=0.3, A_minus=0.2),
        num_pre=3,
        num_post=2,
    )

    rule.step(
        torch.zeros(3),
        post_spike=True,
        post_activity=torch.tensor([1.0, 0.5]),
        timestep=0,
    )
    update = rule.step(
        torch.tensor([1.0, 0.0, 1.0]),
        post_spike=False,
        post_activity=torch.zeros(2),
        timestep=1,
    )

    assert float(update[0, 0].item()) < 0.0
    assert float(update[2, 0].item()) < 0.0
    assert torch.all(update <= 0.0)


def test_time_constant_effects() -> None:
    near_rule = STDPRule(STDPConfig(tau_plus=5.0, tau_minus=20.0, A_plus=0.3, A_minus=0.2), num_pre=1, num_post=1)
    far_rule = STDPRule(STDPConfig(tau_plus=5.0, tau_minus=20.0, A_plus=0.3, A_minus=0.2), num_pre=1, num_post=1)

    near_rule.observe_pre_spikes(torch.tensor([1.0]), timestep=0)
    near_update = near_rule.step(
        torch.zeros(1),
        post_spike=True,
        post_activity=torch.tensor([1.0]),
        timestep=1,
    )

    far_rule.observe_pre_spikes(torch.tensor([1.0]), timestep=0)
    far_update = far_rule.step(
        torch.zeros(1),
        post_spike=True,
        post_activity=torch.tensor([1.0]),
        timestep=5,
    )

    assert float(near_update.abs().item()) > float(far_update.abs().item())


def test_stdp_integration_with_ccc_pool() -> None:
    stdp_pool = CCCPool(
        make_ccc_config(stdp=STDPConfig(tau_plus=12.0, tau_minus=24.0, A_plus=0.15, A_minus=0.05)),
        make_margin_config(),
    )
    baseline_pool = CCCPool(make_ccc_config(stdp=None), make_margin_config())

    for pool in (stdp_pool, baseline_pool):
        for ccc in pool.cccs:
            configure_identity_ccc(ccc)
        commit_ccc(pool.cccs[0])

    assert stdp_pool.cccs[0](shifted_input(), timestep=0).fired is False
    assert baseline_pool.cccs[0](shifted_input(), timestep=0).fired is False

    stdp_before = stdp_pool.cccs[0].feedback_weights.clone()
    baseline_before = baseline_pool.cccs[0].feedback_weights.clone()

    stdp_output = stdp_pool(concept_input(), timestep=1)
    baseline_output = baseline_pool(concept_input(), timestep=1)

    stdp_delta = torch.norm(stdp_pool.cccs[0].feedback_weights - stdp_before).item()
    baseline_delta = torch.norm(baseline_pool.cccs[0].feedback_weights - baseline_before).item()

    assert 0 in stdp_output.fired_indices
    assert 0 in baseline_output.fired_indices
    assert stdp_delta > baseline_delta + 1e-5
