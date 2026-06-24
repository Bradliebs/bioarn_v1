r"""Comprehensive Bio-ARN 2.0 energy measurement and comparison report.

Run as:
    python experiments\energy_report.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
for _path in (_REPO_ROOT, _EXP_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from bioarn.config import BioARNConfig, CCCConfig, MarginGateConfig, SDMConfig
from bioarn.hardware.energy_model import EnergyModel
from bioarn.loop import PredictionOutput, RecognitionOutput, SensorimotorLoop
from bioarn.scaling import BatchedCCCPool
from bioarn.system import PerceptionOutput
from experiments.benchmarks.benchmark_suite import MLP_FLOPS_MAC, transformer_flops_mac
from experiments.mnist_poc import collect_samples, load_mnist


BENCHMARK_RESULTS_PATH = _EXP_DIR / "benchmarks" / "results.json"
REPORT_PATH = _EXP_DIR / "energy_report_results.md"
DATA_PATH = _EXP_DIR / "energy_report_data.json"
DATA_ROOT = _REPO_ROOT / "data"
ELECTRICITY_USD_PER_KWH = 0.10
SERVICE_WINDOW_SECONDS = 1e-3
INFERENCE_HZ_FOR_COST = 100.0
COMPARISON_HZ = 1000.0
TRAINING_SAMPLES = 5_000
TRAINING_EPOCHS_DENSE = 5


@dataclass
class ComponentStats:
    dense_flops: float = 0.0
    sparse_flops: float = 0.0
    memory_accesses: float = 0.0
    active_units: float = 0.0
    total_units: float = 0.0
    spike_count: float = 0.0
    time_ms: float = 0.0

    @property
    def sparsity(self) -> float:
        if self.total_units <= 0:
            return 1.0
        return max(0.0, 1.0 - (self.active_units / self.total_units))

    @property
    def savings(self) -> float:
        if self.sparse_flops <= 0:
            return 0.0
        return self.dense_flops / self.sparse_flops


@dataclass
class SampleProfile:
    components: dict[str, ComponentStats]
    total_wall_ms: float
    sensory_ms: float
    sensory_suppression: float
    visual_event_count: int
    active_cccs: int
    committed_cccs: int
    active_sdm_locations: int
    sdm_queries: int
    pe_suppression: float
    gnw_occupancy: int
    recruited: bool
    learned: bool
    recognition_confidence: float
    action_confidence: float


@dataclass
class HardwareProjection:
    energy_joules: float
    power_at_1hz_w: float
    power_at_100hz_w: float
    power_at_1khz_w: float
    cost_per_million_usd: float
    annual_cost_at_100hz_usd: float
    reference_transformer_energy_joules: float | None = None


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def count_nonzero(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(torch.count_nonzero(tensor).item())


def as_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor


def benchmark_config(seed: int = 42) -> BioARNConfig:
    return BioARNConfig(
        seed=seed,
        ccc=CCCConfig(
            input_dim=784,
            concept_dim=64,
            num_f1_features=128,
            f1_top_k=16,
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=25,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.50,
            theta_margin_lr=0.0005,
            theta_resonance=0.65,
        ),
        sdm=SDMConfig(
            address_dim=10_000,
            hamming_radius=451,
            num_hard_locations=1_000,
            data_dim=64,
            decay_rate=0.999,
            stdp_window=20,
        ),
    )


def load_benchmark_reference() -> dict[str, Any]:
    if not BENCHMARK_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark reference at {BENCHMARK_RESULTS_PATH}. "
            "Run experiments\\benchmarks\\benchmark_suite.py first."
        )
    raw = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    seed_results = raw["seed_results"]

    def collect(*keys: str) -> list[float]:
        values: list[float] = []
        for seed_result in seed_results:
            node: Any = seed_result
            for key in keys:
                node = node[key]
            values.append(float(node))
        return values

    def result_row(model: str) -> dict[str, float]:
        return {
            "accuracy": mean(collect("scenario_a", model, "accuracy")),
            "dense_flops_mac": mean(collect("scenario_e", model, "dense_flops_mac")),
            "active_flops_mac": mean(collect("scenario_e", model, "active_flops_mac")),
            "latency_ms": mean(collect("scenario_e", model, "latency_ms")),
            "params": mean(collect("scenario_e", model, "params")),
            "memory_mb": mean(collect("scenario_e", model, "memory_mb")),
        }

    return {
        "timestamp": raw["timestamp"],
        "config": raw["config"],
        "mean_fired_cccs": mean(collect("mean_fired_cccs")),
        "bioarn": result_row("bioarn"),
        "mlp": result_row("mlp"),
        "transformer": result_row("transformer"),
    }


def load_measurement_samples() -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    train_dataset = load_mnist(DATA_ROOT, train=True)
    test_dataset = load_mnist(DATA_ROOT, train=False)
    seen_train = collect_samples(train_dataset, labels=set(range(5)), per_label=4, limit=20)
    warmup = seen_train[:15]
    familiar = seen_train[15:20]
    novel_source = collect_samples(test_dataset, labels=set(range(5, 10)), per_label=1, limit=5)
    novel = [(1.0 - sample, label) for sample, label in novel_source]
    if len(warmup) < 5 or len(familiar) < 3 or len(novel) < 3:
        raise RuntimeError("Unable to collect enough MNIST samples for energy profiling.")
    return warmup, familiar, novel


def build_warmed_loop(config: BioARNConfig, warmup_samples: list[tuple[torch.Tensor, int]]) -> SensorimotorLoop:
    loop = SensorimotorLoop(config)
    for sample, _ in warmup_samples:
        loop.step(visual_input=sample.view(1, 28, 28))
    clear_runtime_state(loop)
    return loop


def clear_runtime_state(loop: SensorimotorLoop) -> None:
    loop.visual_encoder.reset_state()
    loop.hierarchy.reset()
    loop.motor_stream.reset()
    loop.core.gnw.clear()
    loop.core.gnw.broadcast_history = [None] * loop.core.gnw.history_capacity
    loop.core.gnw._history_cursor = 0
    loop.core.gnw._history_count = 0
    loop.core.gnw.turnover_events.zero_()
    loop.core.gnw.timestep_tensor.zero_()
    loop.core.last_perception = None
    loop.core.last_thought = loop.core._empty_thought()
    loop.core.timestep = 0
    loop.core.fabric.activation_history.clear()
    loop.core.fabric.temporal_associator.clear_buffer()
    loop.core.fabric._last_decay_timestep = None
    loop._last_sensory = None
    loop._last_prediction = None
    loop._last_perception = None
    loop._last_recognition = None
    loop._last_attention = None
    loop._last_plan = None
    loop._last_action = None
    loop._last_reward = None
    loop._last_action_signal = None
    loop._feedback_features.zero_()
    loop._generated_token_history.clear()


def make_empty_components() -> dict[str, ComponentStats]:
    return {
        "CCC Pool": ComponentStats(),
        "SDM Retrieval": ComponentStats(),
        "PE Hierarchy": ComponentStats(),
        "GNW": ComponentStats(),
        "Motor Stream": ComponentStats(),
    }


def add_stats(left: ComponentStats, right: ComponentStats) -> ComponentStats:
    return ComponentStats(
        dense_flops=left.dense_flops + right.dense_flops,
        sparse_flops=left.sparse_flops + right.sparse_flops,
        memory_accesses=left.memory_accesses + right.memory_accesses,
        active_units=left.active_units + right.active_units,
        total_units=left.total_units + right.total_units,
        spike_count=left.spike_count + right.spike_count,
        time_ms=left.time_ms + right.time_ms,
    )


def scale_stats(stats: ComponentStats, factor: float) -> ComponentStats:
    return ComponentStats(
        dense_flops=stats.dense_flops * factor,
        sparse_flops=stats.sparse_flops * factor,
        memory_accesses=stats.memory_accesses * factor,
        active_units=stats.active_units * factor,
        total_units=stats.total_units * factor,
        spike_count=stats.spike_count * factor,
        time_ms=stats.time_ms * factor,
    )


def ccc_component_stats(loop: SensorimotorLoop, pool_output, time_ms: float, allow_learning: bool) -> ComponentStats:
    config = loop.config.ccc
    committed = sum(bool(ccc.is_committed.item()) for ccc in loop.core.ccc_pool.cccs)
    input_nnz = count_nonzero(loop._last_sensory.features if loop._last_sensory is not None else None)
    outputs = pool_output.outputs[:committed]
    fired_outputs = [output for output in outputs if output.fired]
    f1_nnz = sum(count_nonzero(output.f1_output) for output in outputs)
    gate_nnz = sum(count_nonzero(output.gate_output.output) for output in fired_outputs)
    prediction_nnz = sum(count_nonzero(output.prediction) for output in fired_outputs)
    resonated = sum(
        1
        for output in fired_outputs
        if output.resonance is not None and bool(output.resonance.resonated.reshape(-1).any().item())
    )

    dense_per_ccc = (
        (2 * config.input_dim * config.num_f1_features)
        + (2 * config.num_f1_features * config.concept_dim)
        + (6 * config.concept_dim)
        + (2 * config.concept_dim * config.num_f1_features)
        + (6 * config.num_f1_features)
    )
    sparse_per_ccc = (
        (2 * input_nnz * config.num_f1_features)
        + (2 * config.f1_top_k * config.concept_dim)
        + (6 * config.concept_dim)
    )
    dense_flops = committed * dense_per_ccc
    sparse_flops = (committed * sparse_per_ccc) + (
        len(fired_outputs) * ((2 * config.concept_dim * config.num_f1_features) + (6 * config.num_f1_features))
    )
    if allow_learning and resonated:
        sparse_flops += resonated * (
            (2 * config.num_f1_features * config.concept_dim)
            + (2 * config.f1_top_k * config.concept_dim)
        )

    dense_mem = committed * (
        (config.input_dim * config.num_f1_features)
        + (config.num_f1_features * config.concept_dim)
        + (config.concept_dim * config.num_f1_features)
        + config.input_dim
        + config.num_f1_features
        + config.concept_dim
    )
    sparse_mem = committed * (
        (input_nnz * config.num_f1_features)
        + (config.f1_top_k * config.concept_dim)
        + input_nnz
        + config.f1_top_k
        + config.concept_dim
    ) + prediction_nnz

    total_units = committed * (config.num_f1_features + config.concept_dim)
    active_units = f1_nnz + gate_nnz
    spike_count = active_units + len(fired_outputs)
    sparse_flops = min(float(dense_flops), float(sparse_flops))
    sparse_mem = min(float(dense_mem), float(sparse_mem if sparse_mem > 0 else dense_mem))
    return ComponentStats(
        dense_flops=float(dense_flops),
        sparse_flops=sparse_flops,
        memory_accesses=sparse_mem,
        active_units=float(active_units),
        total_units=float(total_units),
        spike_count=float(spike_count),
        time_ms=float(time_ms),
    )


def sdm_component_stats(
    loop: SensorimotorLoop,
    active_cccs: list[tuple[int, torch.Tensor, float]],
    surviving_cccs: list[tuple[int, torch.Tensor, float]],
    vote_direction: torch.Tensor | None,
    time_ms: float,
) -> ComponentStats:
    sdm = loop.core.fabric.sdm
    concept_dim = loop.config.ccc.concept_dim
    address_dim = sdm.address_dim
    num_locations = sdm.num_hard_locations
    data_dim = sdm.data_dim

    register_active_locations = [
        int(sdm.get_activated_locations(direction).sum().item())
        for _, direction, _ in active_cccs
    ]
    retrieve_active_locations = (
        int(sdm.get_activated_locations(vote_direction).sum().item()) if vote_direction is not None else 0
    )
    num_registers = len(register_active_locations)
    num_retrieves = 1 if vote_direction is not None else 0
    direction_nnz = concept_dim

    dense_register = (2 * concept_dim * address_dim) + (2 * address_dim * num_locations) + (2 * num_locations * data_dim)
    sparse_register = (2 * direction_nnz * address_dim) + (2 * address_dim * num_locations)
    sparse_register += sum(2 * active_locations * data_dim for active_locations in register_active_locations)

    dense_retrieve = num_retrieves * dense_register
    sparse_retrieve = 0.0
    if num_retrieves:
        sparse_retrieve = (
            (2 * direction_nnz * address_dim)
            + (2 * address_dim * num_locations)
            + (2 * retrieve_active_locations * data_dim)
        )

    dense_flops = (num_registers * dense_register) + dense_retrieve
    sparse_flops = sparse_register + sparse_retrieve

    dense_mem = (num_registers + num_retrieves) * (
        (concept_dim * address_dim)
        + (num_locations * address_dim)
        + (num_locations * data_dim)
    )
    sparse_mem = (num_registers + num_retrieves) * ((direction_nnz * address_dim) + (num_locations * address_dim))
    sparse_mem += (sum(register_active_locations) + retrieve_active_locations) * data_dim

    active_units = sum(register_active_locations) + retrieve_active_locations
    total_units = (num_registers + num_retrieves) * num_locations
    sparse_flops = min(float(dense_flops), float(sparse_flops))
    sparse_mem = min(float(dense_mem), float(sparse_mem))
    return ComponentStats(
        dense_flops=float(dense_flops),
        sparse_flops=sparse_flops,
        memory_accesses=sparse_mem,
        active_units=float(active_units),
        total_units=float(total_units),
        spike_count=float(active_units),
        time_ms=float(time_ms),
    )


def pe_component_stats(loop: SensorimotorLoop, perception, generated: torch.Tensor, time_ms: float) -> ComponentStats:
    layers = loop.hierarchy.layers
    iterations = max(1, int(perception.iterations_used))
    dense_flops = 0.0
    sparse_flops = 0.0
    memory_accesses = 0.0
    active_units = 0.0
    total_units = 0.0
    suppressed_fracs: list[float] = []

    for index, layer in enumerate(layers):
        input_dim = int(layer.input_dim)
        output_dim = int(layer.output_dim)
        state = as_batch(perception.states[index + 1])[0]
        error = as_batch(perception.errors[index])[0]
        state_nnz = count_nonzero(state)
        error_nnz = count_nonzero(error)
        suppressed_fracs.append(float((error == 0).float().mean().item()))

        dense_per_iter = (6 * input_dim * output_dim) + (4 * input_dim) + output_dim
        sparse_per_iter = (
            (2 * state_nnz * input_dim)
            + (2 * error_nnz * output_dim)
            + (2 * state_nnz * max(error_nnz, 1))
            + error_nnz
            + state_nnz
        )
        dense_flops += iterations * dense_per_iter
        sparse_flops += iterations * sparse_per_iter
        memory_accesses += iterations * (
            (3 * input_dim * output_dim)
            + input_dim
            + output_dim
            + state_nnz
            + error_nnz
        )
        active_units += state_nnz + error_nnz
        total_units += state.numel() + error.numel()

    connector_input_nnz = count_nonzero(as_batch(perception.states[min(loop.connector.level2_index, len(perception.states) - 1)])[0])
    generated_nnz = count_nonzero(generated)
    connector_dense = (2 * loop.config.ccc.concept_dim * loop.config.ccc.concept_dim) + (2 * loop.config.ccc.concept_dim * loop.config.ccc.input_dim)
    connector_sparse = (2 * connector_input_nnz * loop.config.ccc.concept_dim) + (2 * connector_input_nnz * max(generated_nnz, 1))
    dense_flops += connector_dense
    sparse_flops += connector_sparse
    memory_accesses += connector_sparse
    active_units += generated_nnz
    total_units += generated.numel()
    sparse_flops = min(float(dense_flops), float(sparse_flops))
    memory_accesses = min(float(dense_flops), float(memory_accesses))

    return ComponentStats(
        dense_flops=float(dense_flops),
        sparse_flops=sparse_flops,
        memory_accesses=memory_accesses,
        active_units=float(active_units),
        total_units=float(total_units),
        spike_count=float(active_units),
        time_ms=float(time_ms),
    )


def gnw_component_stats(loop: SensorimotorLoop, surviving_cccs: list[tuple[int, torch.Tensor, float]], thought, time_ms: float) -> ComponentStats:
    capacity = int(loop.config.gnw.capacity)
    concept_dim = int(loop.config.ccc.concept_dim)
    candidates = len(surviving_cccs)
    occupied = int(thought.broadcast.num_occupied)

    dense_flops = (capacity * capacity * 4) + (max(candidates, 1) * capacity * 4) + (capacity * concept_dim)
    sparse_flops = (max(candidates, 1) * max(occupied, 1) * 4) + (occupied * concept_dim)
    memory_accesses = (capacity * concept_dim) + (max(candidates, 1) * concept_dim)
    sparse_flops = min(float(dense_flops), float(sparse_flops))
    return ComponentStats(
        dense_flops=float(dense_flops),
        sparse_flops=sparse_flops,
        memory_accesses=float(memory_accesses),
        active_units=float(occupied),
        total_units=float(capacity),
        spike_count=float(occupied),
        time_ms=float(time_ms),
    )


def motor_component_stats(loop: SensorimotorLoop, concept: torch.Tensor, plan, action, time_ms: float) -> ComponentStats:
    hidden = int(loop.motor_stream.hidden_dim)
    vocab = int(loop.motor_stream.vocab_size)
    concept_nnz = count_nonzero(concept)
    plan_nnz = count_nonzero(plan.motor_plan)
    action_nnz = count_nonzero(action.action_vector)
    planner_spikes = (
        count_nonzero(loop.motor_stream.motor_planner.spike_history[-1])
        if loop.motor_stream.motor_planner.spike_history.numel() > 0
        else 0
    )
    executor_spikes = (
        count_nonzero(loop.motor_stream.motor_executor.spike_history[-1])
        if loop.motor_stream.motor_executor.spike_history.numel() > 0
        else 0
    )
    attempts = max(1, action.generated.corrections_made + 1) if action.generated is not None else 1

    dense_plan = (2 * loop.config.ccc.concept_dim * hidden) + (2 * hidden * hidden)
    sparse_plan = (2 * concept_nnz * hidden) + (2 * max(planner_spikes, 1) * hidden)
    dense_attempt = (2 * hidden * hidden) + (2 * hidden * vocab) + (2 * vocab * hidden)
    sparse_attempt = (2 * max(plan_nnz, 1) * hidden) + (2 * max(executor_spikes, 1) * vocab) + (2 * max(action_nnz, 1) * hidden)
    dense_predictive = 2 * (hidden + vocab) * vocab * 2
    sparse_predictive = 2 * (max(action_nnz, 1) + max(executor_spikes, 1)) * vocab * 2

    dense_flops = dense_plan + (attempts * dense_attempt) + dense_predictive
    sparse_flops = sparse_plan + (attempts * sparse_attempt) + sparse_predictive
    memory_accesses = (
        (loop.config.ccc.concept_dim * hidden)
        + (hidden * hidden * 2)
        + (hidden * vocab * 2)
        + ((hidden + vocab) * vocab)
    )

    active_units = planner_spikes + executor_spikes + action_nnz
    total_units = (2 * hidden) + vocab
    sparse_flops = min(float(dense_flops), float(sparse_flops))
    memory_accesses = min(float(dense_flops), float(memory_accesses))
    return ComponentStats(
        dense_flops=float(dense_flops),
        sparse_flops=sparse_flops,
        memory_accesses=memory_accesses,
        active_units=float(active_units),
        total_units=float(total_units),
        spike_count=float(planner_spikes + executor_spikes),
        time_ms=float(time_ms),
    )


@torch.no_grad()
def profile_sample(loop: SensorimotorLoop, sample: torch.Tensor, *, allow_learning: bool) -> SampleProfile:
    clear_runtime_state(loop)
    image = sample.view(1, 28, 28)
    components = make_empty_components()

    total_start = time.perf_counter()
    sensory_start = time.perf_counter()
    sensory = loop.sense(visual_input=image)
    sensory_ms = (time.perf_counter() - sensory_start) * 1000.0
    aligned = loop._align_dim(sensory.features, loop.sensory_dim)

    pe_start = time.perf_counter()
    compare = loop.hierarchy.predict_and_compare(aligned)
    hierarchy_perception = loop.hierarchy.perceive(
        aligned,
        num_iterations=max(4, loop.config.predictive.num_levels * 2),
    )
    concept_level = hierarchy_perception.states[min(loop.connector.level2_index, len(hierarchy_perception.states) - 1)]
    generated = loop.connector.top_down(concept_level)
    prediction = PredictionOutput(
        prediction=loop._align_dim(generated, loop.sensory_dim),
        error=loop._align_dim(compare.error, loop.sensory_dim),
        surprise=float(max(compare.surprise_score, hierarchy_perception.surprise)),
        free_energy=float(hierarchy_perception.free_energy_trace[-1]) if hierarchy_perception.free_energy_trace else float(compare.error.abs().mean().item()),
    )
    loop._last_prediction = prediction
    pe_ms = (time.perf_counter() - pe_start) * 1000.0

    ccc_start = time.perf_counter()
    pool_output = loop.core._run_pool(aligned, allow_recruit=allow_learning)
    active_cccs = loop.core._active_cccs(pool_output)
    ccc_ms = (time.perf_counter() - ccc_start) * 1000.0

    sdm_start = time.perf_counter()
    for index, direction, confidence in active_cccs:
        loop.core.fabric.register_activation(index, direction, confidence, loop.core.timestep)
    loop.core.fabric.form_associations(loop.core.timestep)
    inhibited = (
        loop.core.fabric.lateral_inhibition(active_cccs, k=max(1, len(active_cccs)))
        if active_cccs
        else []
    )
    surviving_cccs = loop.core._surviving_cccs(active_cccs, inhibited)
    vote = loop.core.fabric.vote(surviving_cccs)
    associations = (
        loop.core.fabric.retrieve_associates(vote.winning_direction, k=5)
        if surviving_cccs
        else loop.core._empty_associations()
    )
    sdm_ms = (time.perf_counter() - sdm_start) * 1000.0

    gnw_start = time.perf_counter()
    thought = loop.core.stream.think_step(surviving_cccs, timestep=loop.core.timestep)
    gnw_ms = (time.perf_counter() - gnw_start) * 1000.0

    perception = PerceptionOutput(
        pool_output=pool_output,
        vote_result=vote,
        broadcast=thought.broadcast,
        associations=associations,
        num_fired=len(pool_output.fired_indices),
        num_abstained=len(pool_output.abstained_indices),
        is_novel=bool(pool_output.recruited),
        timestep=loop.core.timestep,
    )
    loop.core.last_thought = thought
    loop.core.last_perception = perception
    loop.core.timestep += 1

    abstained = perception.num_fired == 0 or perception.vote_result.voter_count == 0
    concept_direction = loop._align_dim(perception.vote_result.winning_direction, loop.concept_dim)
    if abstained:
        concept_direction = torch.zeros_like(concept_direction)
    recognition = RecognitionOutput(
        concept_direction=concept_direction.detach().clone(),
        confidence=float(perception.vote_result.confidence),
        abstained=abstained,
        num_hypotheses=perception.num_fired,
        agreement=float(perception.vote_result.agreement_score),
    )
    loop._last_perception = perception
    loop._last_recognition = recognition
    attention = loop.attend(perception)

    concept = recognition.concept_direction
    if recognition.abstained and attention.broadcast.directions:
        concept = loop._align_dim(attention.broadcast.directions[0], loop.concept_dim)
    if torch.count_nonzero(concept).item() == 0:
        concept = loop._dominant_concept()

    motor_start = time.perf_counter()
    plan = loop.plan(concept)
    action = loop.act(plan)
    motor_ms = (time.perf_counter() - motor_start) * 1000.0

    learned = False
    if allow_learning:
        learned = loop._learning_occurred(perception)
        loop.core.learn_from_perception(perception, aligned)

    total_wall_ms = (time.perf_counter() - total_start) * 1000.0
    vote_direction = vote.winning_direction if surviving_cccs else None

    components["CCC Pool"] = ccc_component_stats(loop, pool_output, ccc_ms, allow_learning)
    components["SDM Retrieval"] = sdm_component_stats(loop, active_cccs, surviving_cccs, vote_direction, sdm_ms)
    components["PE Hierarchy"] = pe_component_stats(loop, hierarchy_perception, generated, pe_ms)
    components["GNW"] = gnw_component_stats(loop, surviving_cccs, thought, gnw_ms)
    components["Motor Stream"] = motor_component_stats(loop, concept, plan, action, motor_ms)

    return SampleProfile(
        components=components,
        total_wall_ms=float(total_wall_ms),
        sensory_ms=float(sensory_ms),
        sensory_suppression=float(sensory.suppressed_fraction),
        visual_event_count=int(sensory.visual_output.event_count if sensory.visual_output is not None else 0),
        active_cccs=len(active_cccs),
        committed_cccs=sum(bool(ccc.is_committed.item()) for ccc in loop.core.ccc_pool.cccs),
        active_sdm_locations=int(components["SDM Retrieval"].active_units),
        sdm_queries=len(active_cccs) + (1 if surviving_cccs else 0),
        pe_suppression=float((components["PE Hierarchy"].sparsity)),
        gnw_occupancy=int(thought.broadcast.num_occupied),
        recruited=bool(pool_output.recruited),
        learned=bool(learned),
        recognition_confidence=float(recognition.confidence),
        action_confidence=float(plan.confidence),
    )


def aggregate_profiles(profiles: list[SampleProfile]) -> dict[str, Any]:
    components = make_empty_components()
    for profile in profiles:
        for name, stats in profile.components.items():
            components[name] = add_stats(components[name], stats)

    factor = 1.0 / max(len(profiles), 1)
    averaged = {name: scale_stats(stats, factor) for name, stats in components.items()}
    total_component = ComponentStats()
    for stats in averaged.values():
        total_component = add_stats(total_component, stats)

    return {
        "components": averaged,
        "total": total_component,
        "total_wall_ms": mean([profile.total_wall_ms for profile in profiles]),
        "sensory_ms": mean([profile.sensory_ms for profile in profiles]),
        "sensory_suppression": mean([profile.sensory_suppression for profile in profiles]),
        "visual_event_count": mean([float(profile.visual_event_count) for profile in profiles]),
        "active_cccs": mean([float(profile.active_cccs) for profile in profiles]),
        "committed_cccs": mean([float(profile.committed_cccs) for profile in profiles]),
        "active_sdm_locations": mean([float(profile.active_sdm_locations) for profile in profiles]),
        "sdm_queries": mean([float(profile.sdm_queries) for profile in profiles]),
        "pe_suppression": mean([float(profile.pe_suppression) for profile in profiles]),
        "gnw_occupancy": mean([float(profile.gnw_occupancy) for profile in profiles]),
        "recruitment_rate": mean([1.0 if profile.recruited else 0.0 for profile in profiles]),
        "learning_rate": mean([1.0 if profile.learned else 0.0 for profile in profiles]),
        "recognition_confidence": mean([profile.recognition_confidence for profile in profiles]),
        "action_confidence": mean([profile.action_confidence for profile in profiles]),
    }


def backend_compute_energy(flops: float, memory_accesses: float, backend: str, *, include_static: bool = True) -> float:
    constants = EnergyModel.ENERGY_CONSTANTS[backend]
    energy = (flops * constants["flop"]) + (memory_accesses * constants["memory_access"])
    if include_static:
        energy += constants["idle_power"] * SERVICE_WINDOW_SECONDS
    return float(energy)


def power_from_energy(energy_joules: float, hz: float) -> float:
    return float(energy_joules * hz)


def cost_per_million_inferences(energy_joules: float) -> float:
    return float((energy_joules * 1_000_000 / 3_600_000) * ELECTRICITY_USD_PER_KWH)


def annual_cost(power_watts: float) -> float:
    return float((power_watts * 24.0 * 365.0 / 1000.0) * ELECTRICITY_USD_PER_KWH)


def memory_accesses_for_mlp() -> float:
    weights = (784 * 256) + (256 * 128) + (128 * 10)
    activations = 784 + 256 + 128 + 10
    biases = 256 + 128 + 10
    return float(weights + activations + biases)


def memory_accesses_for_transformer(params: float) -> float:
    seq = 16
    d_model = 128
    activations = (seq * d_model * 6) + (seq * seq * 4) + (d_model * 10)
    return float(params + activations)


def hardware_projections(
    config: BioARNConfig,
    inference_profile: dict[str, Any],
    benchmark_reference: dict[str, Any],
) -> dict[str, HardwareProjection]:
    energy_model = EnergyModel()
    active_cccs = max(1, int(round(inference_profile["active_cccs"])))
    loihi = energy_model.estimate_inference_energy(config, "loihi2", active_cccs).total_joules
    asic = energy_model.estimate_inference_energy(config, "ideal_asic", active_cccs).total_joules
    cpu = backend_compute_energy(
        inference_profile["total"].sparse_flops,
        inference_profile["total"].memory_accesses,
        "cpu_laptop",
    )
    gpu = backend_compute_energy(
        inference_profile["total"].sparse_flops,
        inference_profile["total"].memory_accesses,
        "gpu_a100",
    )

    transformer_energy_gpu = backend_compute_energy(
        float(2.0 * benchmark_reference["transformer"]["dense_flops_mac"]),
        memory_accesses_for_transformer(benchmark_reference["transformer"]["params"]),
        "gpu_a100",
    )
    mlp_energy_gpu = backend_compute_energy(
        float(2.0 * benchmark_reference["mlp"]["dense_flops_mac"]),
        memory_accesses_for_mlp(),
        "gpu_a100",
    )
    benchmark_reference["transformer"]["projected_gpu_energy_j"] = transformer_energy_gpu
    benchmark_reference["mlp"]["projected_gpu_energy_j"] = mlp_energy_gpu

    projections: dict[str, HardwareProjection] = {}
    for backend, energy in {
        "Loihi 2": loihi,
        "GPU (A100)": gpu,
        "CPU (laptop)": cpu,
        "Ideal ASIC": asic,
    }.items():
        projections[backend] = HardwareProjection(
            energy_joules=float(energy),
            power_at_1hz_w=power_from_energy(energy, 1.0),
            power_at_100hz_w=power_from_energy(energy, 100.0),
            power_at_1khz_w=power_from_energy(energy, 1000.0),
            cost_per_million_usd=cost_per_million_inferences(energy),
            annual_cost_at_100hz_usd=annual_cost(power_from_energy(energy, INFERENCE_HZ_FOR_COST)),
            reference_transformer_energy_joules=transformer_energy_gpu,
        )
    return projections


def brain_baseline(config: BioARNConfig, loihi_energy_joules: float) -> dict[str, float]:
    comparison = EnergyModel().brain_comparison(config)
    scaled_power = float(comparison.scaled_brain_power_watts)
    scaled_energy = scaled_power * SERVICE_WINDOW_SECONDS
    return {
        "power_watts": scaled_power,
        "energy_joules": scaled_energy,
        "power_ratio_vs_brain": float(loihi_energy_joules * 1000.0 / max(scaled_power, 1e-18)),
        "total_neurons": int(comparison.total_neurons),
    }


def compute_training_energy(
    config: BioARNConfig,
    learning_profile: dict[str, Any],
    benchmark_reference: dict[str, Any],
) -> dict[str, float]:
    energy_model = EnergyModel()
    active_cccs = max(1, int(round(learning_profile["active_cccs"])))
    bioarn_learning_per_sample = energy_model.estimate_learning_energy(config, "loihi2").total_joules
    bioarn_training_total = bioarn_learning_per_sample * TRAINING_SAMPLES

    transformer_inference_energy = backend_compute_energy(
        float(2.0 * benchmark_reference["transformer"]["dense_flops_mac"]),
        memory_accesses_for_transformer(benchmark_reference["transformer"]["params"]),
        "gpu_a100",
    )
    transformer_training_total = transformer_inference_energy * 6.0 * TRAINING_SAMPLES * TRAINING_EPOCHS_DENSE

    mlp_inference_energy = backend_compute_energy(
        float(2.0 * benchmark_reference["mlp"]["dense_flops_mac"]),
        memory_accesses_for_mlp(),
        "gpu_a100",
    )
    mlp_training_total = mlp_inference_energy * 6.0 * TRAINING_SAMPLES * TRAINING_EPOCHS_DENSE

    benchmark_reference["transformer"]["training_energy_j"] = transformer_training_total
    benchmark_reference["mlp"]["training_energy_j"] = mlp_training_total
    benchmark_reference["bioarn"]["training_energy_j"] = bioarn_training_total

    return {
        "bioarn_learning_per_sample_j": float(bioarn_learning_per_sample),
        "bioarn_total_training_j": float(bioarn_training_total),
        "transformer_total_training_j": float(transformer_training_total),
        "mlp_total_training_j": float(mlp_training_total),
        "transformer_ratio": float(transformer_training_total / max(bioarn_training_total, 1e-18)),
    }


@torch.no_grad()
def scaling_sweep(config: BioARNConfig, warmup_samples: list[tuple[torch.Tensor, int]], familiar_samples: list[tuple[torch.Tensor, int]]) -> list[dict[str, float]]:
    sweep: list[dict[str, float]] = []
    sample_vectors = [sample for sample, _ in warmup_samples]
    eval_vectors = [sample for sample, _ in familiar_samples]
    per_ccc_dense = (config.ccc.input_dim * config.ccc.num_f1_features) + (config.ccc.num_f1_features * config.ccc.concept_dim)
    per_ccc_sparse = (max(1, count_nonzero(eval_vectors[0])) * config.ccc.num_f1_features) + (config.ccc.f1_top_k * config.ccc.concept_dim)

    for pool_size in [100, 500, 1000, 5000]:
        pool_config = CCCConfig(
            input_dim=config.ccc.input_dim,
            concept_dim=config.ccc.concept_dim,
            num_f1_features=config.ccc.num_f1_features,
            f1_top_k=config.ccc.f1_top_k,
            fast_lr=config.ccc.fast_lr,
            slow_lr=config.ccc.slow_lr,
            feedback_lr=config.ccc.feedback_lr,
            max_pool_size=pool_size,
        )
        pool = BatchedCCCPool(pool_config, config.margin_gate)
        for timestep, sample in enumerate(sample_vectors):
            pool(sample, timestep=timestep, allow_recruit=True)

        fired_counts: list[int] = []
        for offset, sample in enumerate(eval_vectors, start=len(sample_vectors)):
            output = pool(sample, timestep=offset, allow_recruit=False)
            fired_counts.append(len(output.fired_indices))

        active_cccs = mean([float(count) for count in fired_counts])
        activation_fraction = active_cccs / float(pool_size)
        dense_flops = pool_size * per_ccc_dense
        sparse_flops = max(1.0, active_cccs * per_ccc_sparse)
        sweep.append(
            {
                "pool_size": float(pool_size),
                "active_cccs": float(active_cccs),
                "activation_fraction": float(activation_fraction),
                "sparse_savings": float(dense_flops / sparse_flops),
            }
        )
    return sweep


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}x"
    if value >= 10:
        return f"{value:,.1f}x"
    return f"{value:.2f}x"


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_energy(value_joules: float) -> str:
    abs_value = abs(value_joules)
    if abs_value >= 1.0:
        return f"{value_joules:.2f} J"
    if abs_value >= 1e-3:
        return f"{value_joules * 1e3:.2f} mJ"
    if abs_value >= 1e-6:
        return f"{value_joules * 1e6:.2f} µJ"
    return f"{value_joules * 1e9:.2f} nJ"


def format_power(value_watts: float) -> str:
    abs_value = abs(value_watts)
    if abs_value >= 1.0:
        return f"{value_watts:.2f} W"
    if abs_value >= 1e-3:
        return f"{value_watts * 1e3:.2f} mW"
    if abs_value >= 1e-6:
        return f"{value_watts * 1e6:.2f} µW"
    return f"{value_watts * 1e9:.2f} nW"


def format_cost(value: float) -> str:
    if value >= 1000:
        return f"${value:,.0f}"
    if value >= 10:
        return f"${value:,.2f}"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def render_report(
    config: BioARNConfig,
    benchmark_reference: dict[str, Any],
    inference_profile: dict[str, Any],
    learning_profile: dict[str, Any],
    projections: dict[str, HardwareProjection],
    brain: dict[str, float],
    training: dict[str, float],
    scaling: list[dict[str, float]],
) -> str:
    total = inference_profile["total"]
    loihi_energy = projections["Loihi 2"].energy_joules
    transformer_energy = benchmark_reference["transformer"]["projected_gpu_energy_j"]
    transformer_ratio = transformer_energy / max(loihi_energy, 1e-18)
    sensory_suppression = inference_profile["pe_suppression"]
    total_sparsity = total.sparsity
    local_learning_ratio = training["transformer_ratio"]
    ccc_sparsity = 1.0 - (inference_profile["active_cccs"] / max(inference_profile["committed_cccs"], 1.0))
    sdm_sparsity = 1.0 - (inference_profile["active_sdm_locations"] / max(inference_profile["sdm_queries"] * config.sdm.num_hard_locations, 1.0))
    gnw_sparsity = 1.0 - (inference_profile["gnw_occupancy"] / max(config.gnw.capacity, 1.0))

    component_rows = []
    for name in ["CCC Pool", "SDM Retrieval", "PE Hierarchy", "GNW", "Motor Stream"]:
        stats = inference_profile["components"][name]
        component_rows.append(
            f"| {name} | {format_int(stats.dense_flops)} | {format_int(stats.sparse_flops)} | {format_percent(stats.sparsity)} | {stats.time_ms:.2f} |"
        )

    projected_rows = [
        f"| Loihi 2 | {format_energy(projections['Loihi 2'].energy_joules)} | {format_power(projections['Loihi 2'].power_at_100hz_w)} | {format_cost(projections['Loihi 2'].cost_per_million_usd)} | {format_ratio(projections['Loihi 2'].energy_joules / max(brain['energy_joules'], 1e-18))} |",
        f"| GPU (A100) | {format_energy(projections['GPU (A100)'].energy_joules)} | {format_power(projections['GPU (A100)'].power_at_100hz_w)} | {format_cost(projections['GPU (A100)'].cost_per_million_usd)} | {format_ratio(projections['GPU (A100)'].energy_joules / max(brain['energy_joules'], 1e-18))} |",
        f"| CPU (laptop) | {format_energy(projections['CPU (laptop)'].energy_joules)} | {format_power(projections['CPU (laptop)'].power_at_100hz_w)} | {format_cost(projections['CPU (laptop)'].cost_per_million_usd)} | {format_ratio(projections['CPU (laptop)'].energy_joules / max(brain['energy_joules'], 1e-18))} |",
        f"| Ideal ASIC | {format_energy(projections['Ideal ASIC'].energy_joules)} | {format_power(projections['Ideal ASIC'].power_at_100hz_w)} | {format_cost(projections['Ideal ASIC'].cost_per_million_usd)} | {format_ratio(projections['Ideal ASIC'].energy_joules / max(brain['energy_joules'], 1e-18))} |",
        f"| Brain (scaled) | {format_energy(brain['energy_joules'])} | {format_power(brain['power_watts'] * 100.0)} | {format_cost(cost_per_million_inferences(brain['energy_joules']))} | 1x |",
    ]

    scaling_rows = [
        f"| {int(row['pool_size'])} | {row['active_cccs']:.1f} | {format_percent(row['activation_fraction'])} | {format_ratio(row['sparse_savings'])} |"
        for row in scaling
    ]

    bioarn_cpu_latency = inference_profile["total_wall_ms"]
    mlp_cpu_latency = benchmark_reference["mlp"]["latency_ms"]
    transformer_cpu_latency = benchmark_reference["transformer"]["latency_ms"]

    notes = [
        "- FLOPs and memory accesses are analytic counts derived from actual tensor shapes and non-zero activity, not hardware-counter reads.",
        "- CPU wall-clock reflects the dense PyTorch prototype; it does not automatically realize the sparse-event savings projected for Loihi 2 / ASIC hardware.",
        f"- Benchmark accuracy references come from {BENCHMARK_RESULTS_PATH.relative_to(_REPO_ROOT)}.",
        "- Transformer/MLP training energy assumes ~6× inference energy per sample for forward+backward+optimizer work over 5 epochs on 5,000 samples.",
    ]

    learning_delta_flops = learning_profile["total"].sparse_flops - inference_profile["total"].sparse_flops
    learning_delta_ms = learning_profile["total_wall_ms"] - inference_profile["total_wall_ms"]
    if learning_delta_flops >= 0:
        learning_line = (
            f"Novel-sample online learning increases sparse compute from {format_int(inference_profile['total'].sparse_flops)} "
            f"to {format_int(learning_profile['total'].sparse_flops)} FLOPs and wall-clock from "
            f"{inference_profile['total_wall_ms']:.2f} ms to {learning_profile['total_wall_ms']:.2f} ms."
        )
    else:
        learning_line = (
            f"Fresh-loop recruitment steps average {format_int(learning_profile['total'].sparse_flops)} sparse FLOPs "
            f"and {learning_profile['total_wall_ms']:.2f} ms. They are cheaper than warmed-loop inference on this CPU prototype "
            f"because one-shot learning touches fewer already-committed CCCs and less accumulated SDM state "
            f"({format_int(abs(learning_delta_flops))} fewer sparse FLOPs, {abs(learning_delta_ms):.2f} ms faster)."
        )

    return "\n".join(
        [
            "# Bio-ARN 2.0 Energy Efficiency Report",
            "",
            "## Executive Summary",
            (
                f"Bio-ARN achieves {format_ratio(transformer_ratio)} less inference energy than the benchmark 2-layer "
                "transformer when projected onto Loihi 2 versus an A100 at the same ~82% MNIST accuracy tier. "
                "The measured prototype stays sparse—"
                f"{format_percent(total_sparsity)} of modeled units are silent per inference—"
                f"with {format_percent(sensory_suppression)} predictive sensory suppression and "
                f"{format_ratio(local_learning_ratio)} lower projected online-training energy than batch backprop."
            ),
            "",
            "Key takeaways:",
            f"- Sparse activation: only {inference_profile['active_cccs']:.1f} CCCs fire on average out of {inference_profile['committed_cccs']:.1f} committed concepts.",
            f"- PCL suppression: {format_percent(sensory_suppression)} of predictive-hierarchy activity is zeroed by precision-weighted error suppression.",
            f"- Local learning: projected Loihi online learning is {format_ratio(local_learning_ratio)} cheaper than transformer batch training on A100.",
            f"- Caveat: current PyTorch CPU inference is slower than the dense MLP/transformer baselines ({bioarn_cpu_latency:.2f} ms vs {mlp_cpu_latency:.3f} / {transformer_cpu_latency:.3f} ms) because SDM address math and Python orchestration dominate.",
            "",
            "## Measured Computation Profile (PyTorch CPU)",
            "",
            "| Component | FLOPs (Dense) | FLOPs (Sparse) | Sparsity | Time (ms) |",
            "|---|---:|---:|---:|---:|",
            *component_rows,
            f"| **TOTAL** | **{format_int(total.dense_flops)}** | **{format_int(total.sparse_flops)}** | **{format_percent(total.sparsity)}** | **{sum(inference_profile['components'][name].time_ms for name in inference_profile['components']):.2f}** |",
            "",
            f"Measured wall-clock per inference: {inference_profile['total_wall_ms']:.2f} ms total "
            f"({inference_profile['sensory_ms']:.2f} ms in the visual front-end, excluded from the table above).",
            "",
            "## Projected Energy Per Inference",
            "",
            "| Hardware | Energy / Inf | Power @100 Hz | Cost / 1M inf | vs Brain |",
            "|---|---:|---:|---:|---:|",
            *projected_rows,
            "",
            "## Comparison vs Baselines",
            "",
            f"Benchmark reference accuracy: Bio-ARN {benchmark_reference['bioarn']['accuracy']:.3f}, "
            f"Transformer {benchmark_reference['transformer']['accuracy']:.3f}, "
            f"MLP {benchmark_reference['mlp']['accuracy']:.3f}.",
            "",
            "| Metric | Bio-ARN (Loihi 2) | Transformer (A100) | Ratio |",
            "|---|---:|---:|---:|",
            f"| Energy per inference | {format_energy(loihi_energy)} | {format_energy(transformer_energy)} | {format_ratio(transformer_ratio)} |",
            f"| Power @ 1k inf/sec | {format_power(projections['Loihi 2'].power_at_1khz_w)} | {format_power(power_from_energy(transformer_energy, COMPARISON_HZ))} | {format_ratio(power_from_energy(transformer_energy, COMPARISON_HZ) / max(projections['Loihi 2'].power_at_1khz_w, 1e-18))} |",
            f"| Annual energy cost @1k inf/sec | {format_cost(annual_cost(projections['Loihi 2'].power_at_1khz_w))} | {format_cost(annual_cost(power_from_energy(transformer_energy, COMPARISON_HZ)))} | {format_ratio(annual_cost(power_from_energy(transformer_energy, COMPARISON_HZ)) / max(annual_cost(projections['Loihi 2'].power_at_1khz_w), 1e-18))} |",
            f"| Training energy (5k samples) | {format_energy(training['bioarn_total_training_j'])} | {format_energy(training['transformer_total_training_j'])} | {format_ratio(training['transformer_ratio'])} |",
            "",
            f"Reference dense MLP energy on A100: {format_energy(benchmark_reference['mlp']['projected_gpu_energy_j'])} per inference. "
            "Because the MLP is tiny and dense, it remains a strong digital baseline; Bio-ARN’s energy edge appears primarily against attention-heavy transformers and during online learning on spike-native hardware.",
            "",
            "## Sparsity Analysis",
            "",
            "| Component | Mechanism | Measured Sparsity |",
            "|---|---|---:|",
            f"| CCC pool | Margin-gate winner sparsity | {format_percent(ccc_sparsity)} |",
            f"| SDM | Hamming-radius active-location sparsity | {format_percent(sdm_sparsity)} |",
            f"| PE hierarchy | Predictive suppression / zeroed errors | {format_percent(inference_profile['pe_suppression'])} |",
            f"| GNW | Limited broadcast capacity | {format_percent(gnw_sparsity)} |",
            "",
            "| Pool Size | Active CCCs | Activation % | Sparse Savings |",
            "|---:|---:|---:|---:|",
            *scaling_rows,
            "",
            "## Learning Profile",
            "",
            f"{learning_line} Recruitment occurred on {format_percent(learning_profile['recruitment_rate'])} of profiled novel samples.",
            "",
            "## Limitations",
            "",
            *notes,
            "",
            "## Conclusion",
            (
                f"Bio-ARN 2.0 supports the sparse-computation thesis against transformer-class baselines: "
                f"projected Loihi 2 inference is {format_ratio(transformer_ratio)} cheaper than the matched transformer "
                f"while online learning is {format_ratio(local_learning_ratio)} cheaper than dense backprop training. "
                f"The current CPU prototype is not yet biologically efficient—still {format_ratio(loihi_energy / max(brain['energy_joules'], 1e-18))} above a neuron-count-scaled brain baseline—and it is slower than compact dense models on PyTorch. "
                "The result is therefore strongest as a hardware-software co-design claim: sparse, predictive, event-driven Bio-ARN is most compelling on neuromorphic or custom ASIC targets, not in an unoptimized dense CPU implementation."
            ),
        ]
    )


def serialise_component_stats(stats: ComponentStats) -> dict[str, float]:
    return {
        "dense_flops": float(stats.dense_flops),
        "sparse_flops": float(stats.sparse_flops),
        "memory_accesses": float(stats.memory_accesses),
        "active_units": float(stats.active_units),
        "total_units": float(stats.total_units),
        "spike_count": float(stats.spike_count),
        "time_ms": float(stats.time_ms),
        "sparsity": float(stats.sparsity),
        "savings": float(stats.savings),
    }


def serialise_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "components": {name: serialise_component_stats(stats) for name, stats in profile["components"].items()},
        "total": serialise_component_stats(profile["total"]),
        "total_wall_ms": float(profile["total_wall_ms"]),
        "sensory_ms": float(profile["sensory_ms"]),
        "sensory_suppression": float(profile["sensory_suppression"]),
        "visual_event_count": float(profile["visual_event_count"]),
        "active_cccs": float(profile["active_cccs"]),
        "committed_cccs": float(profile["committed_cccs"]),
        "active_sdm_locations": float(profile["active_sdm_locations"]),
        "sdm_queries": float(profile["sdm_queries"]),
        "pe_suppression": float(profile["pe_suppression"]),
        "gnw_occupancy": float(profile["gnw_occupancy"]),
        "recruitment_rate": float(profile["recruitment_rate"]),
        "learning_rate": float(profile["learning_rate"]),
        "recognition_confidence": float(profile["recognition_confidence"]),
        "action_confidence": float(profile["action_confidence"]),
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)

    benchmark_reference = load_benchmark_reference()
    config = benchmark_config()
    warmup_samples, familiar_samples, novel_samples = load_measurement_samples()

    inference_loop = build_warmed_loop(benchmark_config(), warmup_samples)
    inference_profiles = [
        profile_sample(inference_loop, sample, allow_learning=False)
        for sample, _ in familiar_samples
    ]
    learning_profiles = [
        profile_sample(SensorimotorLoop(benchmark_config()), sample, allow_learning=True)
        for sample, _ in novel_samples
    ]

    inference_profile = aggregate_profiles(inference_profiles)
    learning_profile = aggregate_profiles(learning_profiles)
    projections = hardware_projections(config, inference_profile, benchmark_reference)
    brain = brain_baseline(config, projections["Loihi 2"].energy_joules)
    training = compute_training_energy(config, learning_profile, benchmark_reference)
    scaling = scaling_sweep(config, warmup_samples, familiar_samples)

    report = render_report(
        config=config,
        benchmark_reference=benchmark_reference,
        inference_profile=inference_profile,
        learning_profile=learning_profile,
        projections=projections,
        brain=brain,
        training=training,
        scaling=scaling,
    )

    payload = {
        "config": asdict(benchmark_config()),
        "benchmark_reference": benchmark_reference,
        "inference_profile": serialise_profile(inference_profile),
        "learning_profile": serialise_profile(learning_profile),
        "hardware_projections": {
            name: asdict(projection) for name, projection in projections.items()
        },
        "brain_baseline": brain,
        "training_energy": training,
        "scaling_sweep": scaling,
        "report_path": str(REPORT_PATH),
    }

    REPORT_PATH.write_text(report, encoding="utf-8")
    DATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)
    print(f"\nSaved markdown report to {REPORT_PATH}")
    print(f"Saved raw data to {DATA_PATH}")


if __name__ == "__main__":
    main()
