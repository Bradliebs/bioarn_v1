# API Reference

This reference documents the main public classes exposed by Bio-ARN 2.0. Signatures are taken from the live runtime annotations in the current package.

For system-level context, read [Architecture Guide](architecture.md) first.

## `bioarn.core`

Foundational spiking and concept-cell primitives.

### `LIFNeuron`

Leaky integrate-and-fire neuron with persistent membrane state and refractory handling.

**Constructor**: `LIFNeuron(num_neurons: Optional[int] = None, config: Optional[SpikingConfig] = None) -> None`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `num_neurons` | `Optional[int]` | `None` |
| `config` | `Optional[SpikingConfig]` | `None` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `reset_state(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None` | `None` | Clears runtime state so the component can start a fresh episode. |
| `forward(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` | `tuple[torch.Tensor, torch.Tensor]` | Process single-step or multi-step current input. |

### `MarginGate`

Honest-abstention gate that decides whether a concept activation should fire or defer.

**Constructor**: `MarginGate(config: MarginGateConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `MarginGateConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `forward(self, input_activation: torch.Tensor, concept_direction: torch.Tensor) -> MarginGateOutput` | `MarginGateOutput` | Gate activations using a cosine-similarity threshold. |
| `adapt_threshold(self, recent_fire_rate: float) -> None` | `None` | Adapt the margin threshold to keep firing selective. |
| `check_resonance(self, prediction: torch.Tensor, actual_input: torch.Tensor) -> ResonanceOutput` | `ResonanceOutput` | Check whether a prediction resonates strongly enough to learn from. |
| `get_stats(self) -> dict[str, float | int]` | `dict[str, float | int]` | Return running gate statistics as Python scalars. |

### `ConceptCellCluster`

Single concept-cell unit combining sparse feature selection, concept activation, feedback, and local learning.

**Constructor**: `ConceptCellCluster(config: CCCConfig, margin_config: MarginGateConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `CCCConfig` | `—` |
| `margin_config` | `MarginGateConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `empty_output(self, raw_input: torch.Tensor) -> CCCOutput` | `CCCOutput` | Return an abstention placeholder without processing input. |
| `f1_encode(self, raw_input: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Project raw input into sparse F1 feature space. |
| `f2_activate(self, f1_output: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Project F1 activity into concept space. |
| `generate_prediction(self, f2_activation: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Generate a top-down prediction in F1 space. |
| `forward(self, raw_input: torch.Tensor, timestep: int = 0) -> CCCOutput` | `CCCOutput` | Run the full CCC processing pipeline. |
| `learn_fast(self, raw_input: torch.Tensor, f1_output: torch.Tensor) -> None` | `None` | Commit an unassigned CCC to a new concept in one shot. |
| `learn_slow(self, raw_input: torch.Tensor, f1_output: torch.Tensor, resonance: ResonanceOutput) -> None` | `None` | Apply local Hebbian refinement after resonance. |
| `get_info(self) -> dict[str, bool | float | int | dict[str, float | int]]` | `dict[str, bool | float | int | dict[str, float | int]]` | Return CCC status and gate statistics. |

### `CCCPool`

Pool of CCCs that runs competition, concept recruitment, and sparse winner selection.

**Constructor**: `CCCPool(config: CCCConfig, margin_config: MarginGateConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `CCCConfig` | `—` |
| `margin_config` | `MarginGateConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `forward(self, raw_input: torch.Tensor, timestep: int = 0) -> CCCPoolOutput` | `CCCPoolOutput` | Run all committed CCCs and recruit a new one if all abstain. |
| `get_winners(self, pool_output: CCCPoolOutput, k: int = 5) -> list[int]` | `list[int]` | Return the top-k firing CCC indices by confidence. |
| `get_pool_stats(self) -> dict[str, float | int]` | `dict[str, float | int]` | Return high-level pool statistics. |

## `bioarn.memory`

Sparse memory and concept association utilities.

### `SparseDistributedMemory`

Kanerva-style sparse memory for content-addressed read/write and association.

**Constructor**: `SparseDistributedMemory(config: SDMConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `SDMConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `compute_address(self, concept_direction: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Convert a continuous concept direction vector to a binary SDM address. |
| `get_activated_locations(self, address: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Return hard locations within the configured Hamming radius. |
| `write(self, address: torch.Tensor, data: torch.Tensor) -> None` | `None` | Store data at all hard locations activated by the address. |
| `read(self, address: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Retrieve data by summing over activated hard locations. |
| `associate(self, address_a: torch.Tensor, address_b: torch.Tensor, data_a: torch.Tensor, data_b: torch.Tensor, temporal_order: bool = True) -> None` | `None` | Store a bidirectional association between two addressable patterns. |
| `retrieve_associates(self, cue_address: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Retrieve the strongest associated data from a partial or noisy cue. |
| `inhibit(self, activations: torch.Tensor, addresses: torch.Tensor, k: int) -> torch.Tensor` | `torch.Tensor` | Keep the top-k strongest non-overlapping activations. |
| `get_stats(self) -> dict[str, float | int]` | `dict[str, float | int]` | Return basic occupancy and usage statistics. |

### `AssociativeFabric`

Association manager that wraps SDM with temporal binding, voting, and inhibition.

**Constructor**: `AssociativeFabric(sdm_config: SDMConfig, ccc_config: CCCConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `sdm_config` | `SDMConfig` | `—` |
| `ccc_config` | `CCCConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `register_activation(self, ccc_index: int, concept_direction: torch.Tensor, confidence: float, timestep: int) -> None` | `None` | Record a CCC activation and write it into sparse distributed memory. |
| `form_associations(self, timestep: int) -> None` | `None` | Strengthen recent co-activations and temporal causal links. |
| `retrieve_associates(self, cue_direction: torch.Tensor, k: int = 5) -> AssociationResult` | `AssociationResult` | Retrieve the top-k associates for a concept cue. |
| `retrieve_sequence(self, start_direction: torch.Tensor, steps: int = 5) -> list[torch.Tensor]` | `list[torch.Tensor]` | Follow the strongest learned temporal chain from a starting concept. |
| `lateral_inhibition(self, active_cccs: list[tuple[int, torch.Tensor, float]], k: int) -> list[tuple[int, float]]` | `list[tuple[int, float]]` | Keep the strongest non-overlapping active CCCs. |
| `vote(self, active_cccs: list[tuple[int, torch.Tensor, float]]) -> VoteResult` | `VoteResult` | Form a distributed consensus across active CCCs. |
| `get_stats(self) -> dict[str, float | int | tuple[int, int, float] | None]` | `dict[str, float | int | tuple[int, int, float] | None]` | Return high-level sparse-association statistics. |

## `bioarn.predictive`

Predictive-coding layers, stacks, and hierarchy orchestration.

### `PCLayer`

Single predictive-coding layer with local state, precision, and weight updates.

**Constructor**: `PCLayer(input_dim: int, output_dim: int, config: PredictiveConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `input_dim` | `int` | `—` |
| `output_dim` | `int` | `—` |
| `config` | `PredictiveConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `predict(self, higher_state: torch.Tensor | None = None) -> torch.Tensor` | `torch.Tensor` | Generate a prediction for the level below. |
| `compute_error(self, actual_input: torch.Tensor, prediction: torch.Tensor | None = None) -> torch.Tensor` | `torch.Tensor` | Compute precision-weighted and PCL-suppressed prediction errors. |
| `update_state(self, error: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Update the hidden state from bottom-up prediction errors. |
| `update_weights(self, error: torch.Tensor, state: torch.Tensor | None = None) -> None` | `None` | Apply local Hebbian learning and renormalize weight rows. |
| `update_precision(self, error: torch.Tensor) -> None` | `None` | Adapt precision inversely to prediction error magnitude. |
| `forward(self, actual_input: torch.Tensor, higher_state: torch.Tensor | None = None, learn: bool = True) -> PCLayerOutput` | `PCLayerOutput` | Run predict → error → state update → local learning. |
| `reset_state(self) -> None` | `None` | Reset the persistent state for a new episode. |

### `PCStack`

Multi-layer predictive-coding stack used for iterative inference and top-down generation.

**Constructor**: `PCStack(layer_dims: list[int], config: PredictiveConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `layer_dims` | `list[int]` | `—` |
| `config` | `PredictiveConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `forward(self, sensory_input: torch.Tensor, num_iterations: int = 5, learn: bool = True) -> PCStackOutput` | `PCStackOutput` | Iteratively settle the hierarchy toward lower free energy. |
| `generate(self, top_state: torch.Tensor, num_levels: int | None = None) -> torch.Tensor` | `torch.Tensor` | Cascade top-down predictions from the top state. |
| `reset(self) -> None` | `None` | Reset all persistent layer states. |

### `PredictiveHierarchy`

High-level predictive-coding facade for perception, generation, and active inference.

**Constructor**: `PredictiveHierarchy(layer_dims: list[int], config: PredictiveConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `layer_dims` | `list[int]` | `—` |
| `config` | `PredictiveConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `forward(self, sensory_input: torch.Tensor, num_iterations: int = 10) -> HierarchyPerceptionOutput` | `HierarchyPerceptionOutput` | Iteratively settle the hierarchy toward lower free energy. |
| `perceive(self, sensory_input: torch.Tensor, num_iterations: int = 10) -> HierarchyPerceptionOutput` | `HierarchyPerceptionOutput` | Runs the perception path for a new input. |
| `generate(self, top_state: torch.Tensor, num_levels: int | None = None) -> HierarchyGenerationOutput` | `HierarchyGenerationOutput` | Cascade top-down predictions from the top state. |
| `predict_and_compare(self, sensory_input: torch.Tensor) -> PredictionQualityOutput` | `PredictionQualityOutput` | Public method defined on PredictiveHierarchy. |
| `active_inference_step(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> ActionSignal` | `ActionSignal` | Public method defined on PredictiveHierarchy. |
| `get_level_states(self) -> list[torch.Tensor]` | `list[torch.Tensor]` | Public method defined on PredictiveHierarchy. |
| `get_level_errors(self) -> list[torch.Tensor]` | `list[torch.Tensor]` | Public method defined on PredictiveHierarchy. |
| `get_precision_map(self) -> list[torch.Tensor]` | `list[torch.Tensor]` | Public method defined on PredictiveHierarchy. |
| `reset(self) -> None` | `None` | Reset all persistent layer states. |

## `bioarn.workspace`

Attention, broadcast, and thought-stream components.

### `GlobalNeuronalWorkspace`

Competitive working-memory workspace that broadcasts a small set of active concepts.

**Constructor**: `GlobalNeuronalWorkspace(config: GNWConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `GNWConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `compete(self, candidates: list[tuple[int, torch.Tensor, float]]) -> list[int]` | `list[int]` | Run softmax competition and return winning candidate CCC indices. |
| `update(self, fired_cccs: list[tuple[int, torch.Tensor, float]], timestep: int) -> tuple[list[int], list[int]]` | `tuple[list[int], list[int]]` | Advance workspace dynamics and add new competition winners. |
| `broadcast(self) -> BroadcastOutput` | `BroadcastOutput` | Emit the current broadcast signal and record the dominant thought. |
| `get_stream(self, last_n: int = 10) -> list[GNWSlot]` | `list[GNWSlot]` | Return the recent dominant broadcast history in temporal order. |
| `attend(self, query_direction: torch.Tensor) -> AttentionResult` | `AttentionResult` | Attend over current slot occupants using cosine-similarity relevance. |
| `inject(self, ccc_index: int, direction: torch.Tensor, priority: float = 1.0) -> None` | `None` | Force a concept into the workspace for top-down control. |
| `is_full(self) -> bool` | `bool` | Return whether all workspace slots are occupied. |
| `clear(self) -> None` | `None` | Empty the current workspace contents while preserving history. |
| `get_stats(self) -> dict[str, float]` | `dict[str, float]` | Summarize occupancy and turnover statistics. |

### `StreamOfConsciousness`

Serial thought-chain helper built on GNW broadcasts and associative recall.

**Constructor**: `StreamOfConsciousness(gnw: GlobalNeuronalWorkspace, config: GNWConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `gnw` | `GlobalNeuronalWorkspace` | `—` |
| `config` | `GNWConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `think_step(self, fired_cccs: list[tuple[int, torch.Tensor, float]], timestep: int) -> ThoughtOutput` | `ThoughtOutput` | Advance one conscious reasoning step. |
| `get_thought_chain(self, n: int = 5) -> list[torch.Tensor]` | `list[torch.Tensor]` | Return the last N distinct dominant thoughts. |
| `detect_rumination(self) -> bool` | `bool` | Detect repeated fixation on the same dominant concept. |

## `bioarn.sensorimotor`

Perception encoders and motor-generation streams.

### `VisualEncoder`

Event-based visual front end for frames, image deltas, and predictive suppression.

**Constructor**: `VisualEncoder(input_shape: tuple, output_dim: int, config: SpikingConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `input_shape` | `tuple` | `—` |
| `output_dim` | `int` | `—` |
| `config` | `SpikingConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `reset_state(self) -> None` | `None` | Reset recurrent state and predictive buffers. |
| `delta_encode_frame(self, current_frame: torch.Tensor, previous_frame: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Compute signed ON/OFF events between two consecutive frames. |
| `forward(self, frame: torch.Tensor, prev_frame: torch.Tensor | None = None) -> VisionOutput` | `VisionOutput` | Define the computation performed at every call. |
| `encode_sequence(self, frames: torch.Tensor) -> list[VisionOutput]` | `list[VisionOutput]` | Encode a time-first frame sequence of shape (time, batch, ...). |
| `get_suppression_stats(self) -> dict[str, float | int]` | `dict[str, float | int]` | Return cumulative predictive suppression statistics. |

### `LanguageEncoder`

Token-to-spike encoder for language input.

**Constructor**: `LanguageEncoder(vocab_size: int, embedding_dim: int, output_dim: int, config: SpikingConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `vocab_size` | `int` | `—` |
| `embedding_dim` | `int` | `—` |
| `output_dim` | `int` | `—` |
| `config` | `SpikingConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `reset_state(self) -> None` | `None` | Reset temporal spiking state and predictive carry-over. |
| `forward(self, token_ids: torch.Tensor) -> LanguageOutput` | `LanguageOutput` | Define the computation performed at every call. |
| `encode_text(self, text: str, char_to_idx: dict) -> LanguageOutput` | `LanguageOutput` | Convenience wrapper for raw character strings. |

### `LanguageMotorStream`

Concept-to-token motor stream with rollout and self-monitoring support.

**Constructor**: `LanguageMotorStream(concept_dim: int, vocab_size: int, hidden_dim: int = 128, config: SpikingConfig | None = None) -> None`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `concept_dim` | `int` | `—` |
| `vocab_size` | `int` | `—` |
| `hidden_dim` | `int` | `128` |
| `config` | `SpikingConfig | None` | `None` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `plan(self, concept_direction: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Convert a concept direction into a hidden motor plan. |
| `execute_step(self, motor_plan: torch.Tensor) -> MotorStepOutput` | `MotorStepOutput` | Produce one predictive output step from the current motor plan. |
| `self_monitor_step(self, produced: torch.Tensor, predicted: torch.Tensor) -> MonitorOutput` | `MonitorOutput` | Compare produced output against the current top-down prediction. |
| `predict_next(self, current_output: torch.Tensor, motor_state: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Predict the next output distribution from current output and motor state. |
| `generate_sequence(self, concept_direction: torch.Tensor, max_length: int = 50, temperature: float = 1.0) -> GenerationOutput` | `GenerationOutput` | Generate a language sequence through predictive execution and monitoring. |
| `reset(self) -> None` | `None` | Clear internal state for a fresh generation episode. |

## `bioarn.reward`

Novelty, curiosity, and dopamine-style modulation.

### `RewardSystem`

Intrinsic reward, novelty, curiosity, and learning-rate modulation controller.

**Constructor**: `RewardSystem(config: RewardConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `RewardConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `compute_intrinsic_reward(self, current_error: float, previous_error: float) -> RewardSignal` | `RewardSignal` | Reward prediction-error reduction relative to recent experience. |
| `detect_novelty(self, prediction_error: float) -> NoveltySignal` | `NoveltySignal` | Detect large deviations from the running prediction-error baseline. |
| `compute_curiosity(self, available_options: list[float]) -> CuriositySignal` | `CuriositySignal` | Prefer options with high expected learning, not merely high confusion. |
| `apply_external_reward(self, reward_value: float, source: str = task) -> None` | `None` | Inject an external reward signal into the modulation dynamics. |
| `get_modulation(self) -> ModulationOutput` | `ModulationOutput` | Return the current cross-system modulation values. |
| `step(self, prediction_error: float, learned: bool = False) -> RewardStepOutput` | `RewardStepOutput` | Advance reward, novelty, curiosity, and dopamine state by one timestep. |
| `reset(self) -> None` | `None` | Reset all reward, novelty, and curiosity state. |
| `get_stats(self) -> dict[str, object]` | `dict[str, object]` | Expose bounded histories and current modulation state. |

### `DopamineScheduler`

Tonic and phasic dopamine-style scheduler used by the reward system.

**Constructor**: `DopamineScheduler(config: RewardConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `RewardConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `burst(self, magnitude: float) -> None` | `None` | Register a positive phasic dopamine burst. |
| `dip(self, magnitude: float) -> None` | `None` | Register a negative phasic dopamine dip. |
| `tonic_level(self) -> float` | `float` | Return the current effective dopamine level. |
| `update(self) -> None` | `None` | Decay phasic transients and relax tonic dopamine back to baseline. |
| `reset(self) -> None` | `None` | Reset dopamine dynamics to baseline. |

## `bioarn.system`

Core cognition stack entry point.

### `BioARNCore`

Core cognitive system wiring CCCs, memory, GNW, and thought dynamics.

**Constructor**: `BioARNCore(config: BioARNConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `BioARNConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `perceive(self, raw_input: torch.Tensor) -> PerceptionOutput` | `PerceptionOutput` | Run the full external perception loop, including CCC recruitment. |
| `think(self, num_steps: int = 5) -> list[ThoughtOutput]` | `list[ThoughtOutput]` | Run internal association-driven reasoning without new external input. |
| `recognize(self, raw_input: torch.Tensor) -> RecognitionOutput` | `RecognitionOutput` | Recognize an input without recruiting a new CCC for novel patterns. |
| `learn_from_perception(self, perception: PerceptionOutput, raw_input: torch.Tensor) -> None` | `None` | Apply any deferred learning that was not handled during perception. |
| `forward(self, raw_input: torch.Tensor, learn: bool = True) -> BioARNCoreOutput` | `BioARNCoreOutput` | Run one full system step with optional continual learning. |
| `get_system_stats(self) -> dict` | `dict` | Summarize pool, fabric, workspace, and system-level state. |

## `bioarn.loop`

Embodied perception-action loop entry point.

### `SensorimotorLoop`

End-to-end embodied loop that senses, predicts, recognizes, plans, acts, and learns.

**Constructor**: `SensorimotorLoop(config: BioARNConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `BioARNConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `sense(self, visual_input: torch.Tensor | None = None, language_input: torch.Tensor | None = None) -> SensoryOutput` | `SensoryOutput` | Public method defined on SensorimotorLoop. |
| `predict(self, sensory_features: torch.Tensor) -> PredictionOutput` | `PredictionOutput` | Public method defined on SensorimotorLoop. |
| `recognize(self, sensory_features: torch.Tensor) -> RecognitionOutput` | `RecognitionOutput` | Scores the input against known concepts. |
| `attend(self, perception: PerceptionOutput) -> AttentionOutput` | `AttentionOutput` | Public method defined on SensorimotorLoop. |
| `plan(self, concept: torch.Tensor, goal: torch.Tensor | None = None) -> PlanOutput` | `PlanOutput` | Public method defined on SensorimotorLoop. |
| `act(self, plan: PlanOutput) -> ActionOutput` | `ActionOutput` | Public method defined on SensorimotorLoop. |
| `step(self, visual_input: torch.Tensor | None = None, language_input: torch.Tensor | None = None, goal: torch.Tensor | None = None) -> LoopStepOutput` | `LoopStepOutput` | Advances the component by one runtime step. |
| `run(self, inputs: list[torch.Tensor], num_steps: int | None = None, generate: bool = False) -> LoopRunOutput` | `LoopRunOutput` | Public method defined on SensorimotorLoop. |
| `generate_text(self, seed_concept: torch.Tensor, max_tokens: int = 50) -> str` | `str` | Public method defined on SensorimotorLoop. |
| `active_inference_step(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor` | `torch.Tensor` | Public method defined on SensorimotorLoop. |

## `bioarn.scaling`

Optimized variants for scaling studies.

### `ScaledBioARN`

Scaled Bio-ARN variant that swaps in optimized pool and memory components.

**Constructor**: `ScaledBioARN(config: BioARNConfig, use_optimized: bool = True)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `BioARNConfig` | `—` |
| `use_optimized` | `bool` | `True` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `learn_from_perception(self, perception, raw_input: torch.Tensor) -> None` | `None` | Apply any deferred learning that was not handled during perception. |

> `ScaledBioARN` inherits the public perception/recognition/thinking interface from `BioARNCore`; only the optimized learning override is declared directly on the subclass.

### `BatchedCCCPool`

Vectorized CCCPool implementation for larger-scale experiments.

**Constructor**: `BatchedCCCPool(config: CCCConfig, margin_config: MarginGateConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `CCCConfig` | `—` |
| `margin_config` | `MarginGateConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `recruit(self, raw_input: torch.Tensor, timestep: int = 0) -> tuple[int | None, CCCOutput | None]` | `tuple[int | None, CCCOutput | None]` | Public method defined on BatchedCCCPool. |
| `forward(self, raw_input: torch.Tensor, timestep: int = 0, allow_recruit: bool = True) -> CCCPoolOutput` | `CCCPoolOutput` | Define the computation performed at every call. |
| `get_winners(self, pool_output: CCCPoolOutput, k: int = 5) -> list[int]` | `list[int]` | Public method defined on BatchedCCCPool. |
| `get_pool_stats(self) -> dict[str, float | int]` | `dict[str, float | int]` | Returns aggregate pool diagnostics. |
| `load_from_pool(self, pool: "CCCPool | BatchedCCCPool") -> "BatchedCCCPool"` | `BatchedCCCPool` | Public method defined on BatchedCCCPool. |

## `bioarn.hardware`

Backend abstraction and neuromorphic deployment helpers.

### `NeuromorphicBackend`

Abstract backend interface for software or neuromorphic execution targets.

**Constructor**: `NeuromorphicBackend()`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| — | — | — |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `create_neuron_group(self, num_neurons: int, neuron_type: str, params: dict | None = None) -> NeuronGroupHandle` | `NeuronGroupHandle` | Create a hardware-managed neuron population. |
| `create_synapse(self, source: NeuronGroupHandle, target: NeuronGroupHandle, weights: torch.Tensor, learning_rule: str) -> SynapseHandle` | `SynapseHandle` | Create a hardware-managed synaptic projection. |
| `inject_spikes(self, neuron_group: NeuronGroupHandle, spike_pattern: torch.Tensor) -> None` | `None` | Inject a spike tensor into a neuron population. |
| `step(self, num_steps: int = 1) -> StepResult` | `StepResult` | Advance the backend by one or more timesteps. |
| `read_spikes(self, neuron_group: NeuronGroupHandle) -> torch.Tensor` | `torch.Tensor` | Read the latest spike state from a neuron population. |
| `update_weights(self, synapse: SynapseHandle, new_weights: torch.Tensor) -> None` | `None` | Replace the weights of a managed synapse projection. |
| `reset(self) -> None` | `None` | Reset all backend state. |
| `get_energy_estimate(self) -> EnergyEstimate` | `EnergyEstimate` | Return the backend energy estimate for accumulated work. |
| `get_hardware_info(self) -> HardwareInfo` | `HardwareInfo` | Return backend capabilities and coarse hardware constraints. |

### `PyTorchBackend`

Reference backend that simulates spike groups and synapses in PyTorch.

**Constructor**: `PyTorchBackend(*, device: str | torch.device = cpu, dtype: torch.dtype = torch.float32) -> None`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `device` | `str | torch.device` | `cpu` |
| `dtype` | `torch.dtype` | `torch.float32` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `create_neuron_group(self, num_neurons: int, neuron_type: str, params: dict | None = None) -> NeuronGroupHandle` | `NeuronGroupHandle` | Create a hardware-managed neuron population. |
| `create_synapse(self, source: NeuronGroupHandle, target: NeuronGroupHandle, weights: torch.Tensor, learning_rule: str) -> SynapseHandle` | `SynapseHandle` | Create a hardware-managed synaptic projection. |
| `inject_spikes(self, neuron_group: NeuronGroupHandle, spike_pattern: torch.Tensor) -> None` | `None` | Inject a spike tensor into a neuron population. |
| `step(self, num_steps: int = 1) -> StepResult` | `StepResult` | Advance the backend by one or more timesteps. |
| `read_spikes(self, neuron_group: NeuronGroupHandle) -> torch.Tensor` | `torch.Tensor` | Read the latest spike state from a neuron population. |
| `update_weights(self, synapse: SynapseHandle, new_weights: torch.Tensor) -> None` | `None` | Replace the weights of a managed synapse projection. |
| `reset(self) -> None` | `None` | Reset all backend state. |
| `get_energy_estimate(self) -> EnergyEstimate` | `EnergyEstimate` | Return the backend energy estimate for accumulated work. |
| `get_hardware_info(self) -> HardwareInfo` | `HardwareInfo` | Return backend capabilities and coarse hardware constraints. |

### `LoihiMapping`

Structured mapping helper for deploying Bio-ARN abstractions onto Loihi 2 concepts.

**Constructor**: `LoihiMapping(config: BioARNConfig)`

**Constructor parameters**

| Name | Type | Default |
|---|---|---|
| `config` | `BioARNConfig` | `—` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `map_lif_neuron(self) -> LoihiNeuronSpec` | `LoihiNeuronSpec` | Map Bio-ARN LIF parameters onto a Loihi 2 compartment. |
| `map_ccc_to_cores(self) -> LoihiCCCMapping` | `LoihiCCCMapping` | Map one CCC onto Loihi 2 feature, concept, and comparator resources. |
| `map_sdm_to_memory(self) -> LoihiSDMMapping` | `LoihiSDMMapping` | Map SDM hard locations, address matching, and readout onto Loihi 2. |
| `map_pe_to_pipeline(self) -> LoihiPEMapping` | `LoihiPEMapping` | Map predictive coding layers onto Loihi 2 forward/error/learning stages. |
| `map_gnw_to_circuit(self) -> LoihiGNWMapping` | `LoihiGNWMapping` | Map GNW competition, broadcast, and fatigue onto Loihi 2 circuits. |
| `map_full_system(self) -> LoihiSystemMapping` | `LoihiSystemMapping` | Estimate full-system Loihi 2 resource usage for Bio-ARN. |
