"""Closed-loop sensorimotor orchestration for Bio-ARN 2.0."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import BioARNConfig
from bioarn.predictive.hierarchy import ActionSignal, HierarchyConnector, PredictiveHierarchy
from bioarn.reward.novelty import ModulationOutput, NoveltySignal, RewardStepOutput, RewardSystem
from bioarn.sensorimotor.language import LanguageEncoder, LanguageOutput
from bioarn.sensorimotor.motor import ConceptToLanguage, GenerationOutput, LanguageMotorStream
from bioarn.sensorimotor.vision import VisionOutput, VisualEncoder
from bioarn.system import BioARNCore, PerceptionOutput, RecognitionOutput
from bioarn.workspace.gnw import BroadcastOutput


def _as_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0), True
    if tensor.dim() != 2:
        raise ValueError("Expected a 1D or 2D tensor.")
    return tensor, False


def _restore_shape(tensor: torch.Tensor, squeeze: bool) -> torch.Tensor:
    return tensor.squeeze(0) if squeeze and tensor.shape[0] == 1 else tensor


def _match_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1)
    raise ValueError("Batch size mismatch in sensorimotor loop.")


@dataclass
class SensoryOutput:
    """Fused bottom-up sensory state for the current loop step."""

    features: torch.Tensor
    visual_output: VisionOutput | None
    language_output: LanguageOutput | None
    suppressed_fraction: float


@dataclass
class PredictionOutput:
    """Predictive-coding state for the current loop step."""

    prediction: torch.Tensor
    error: torch.Tensor
    surprise: float
    free_energy: float


@dataclass
class AttentionOutput:
    """Workspace broadcast, novelty, and reward modulation state."""

    broadcast: BroadcastOutput
    novelty: NoveltySignal
    modulation: ModulationOutput
    shifted: bool


@dataclass
class PlanOutput:
    """Motor-plan state derived from the current recognized concept."""

    motor_plan: torch.Tensor
    action_signal: ActionSignal | None
    confidence: float


@dataclass
class ActionOutput:
    """Executed action and self-monitoring summary."""

    generated: GenerationOutput | None
    action_vector: torch.Tensor | None
    self_correction: bool
    feedback_error: float


@dataclass
class LoopStepOutput:
    """One full sense → act → reward cycle."""

    sensory: SensoryOutput
    prediction: PredictionOutput
    recognition: RecognitionOutput
    attention: AttentionOutput
    plan: PlanOutput | None
    action: ActionOutput | None
    reward: RewardStepOutput
    learned: bool
    timestep: int


@dataclass
class LoopRunOutput:
    """Aggregate trace over multiple loop steps."""

    steps: list[LoopStepOutput]
    free_energy_trace: list[float]
    reward_trace: list[float]
    novelty_events: list[int]
    generated_text: str | None
    total_learning_events: int
    final_stats: dict


class SensorimotorLoop(nn.Module):
    """Full Bio-ARN embodied loop: sense, predict, recognize, attend, plan, act."""

    def __init__(self, config: BioARNConfig):
        super().__init__()
        torch.manual_seed(int(config.seed))

        self.config = config
        self.sensory_dim = int(config.ccc.input_dim)
        self.concept_dim = int(config.ccc.concept_dim)
        self.visual_input_shape = self._infer_visual_shape(self.sensory_dim)
        self.vocab_size = max(32, min(256, self.concept_dim * 2))
        self.embedding_dim = max(8, min(32, self.concept_dim))
        self.motor_hidden_dim = max(64, min(256, self.concept_dim * 4))

        self.visual_encoder = VisualEncoder(
            input_shape=self.visual_input_shape,
            output_dim=self.sensory_dim,
            config=config.spiking,
        )
        self.language_encoder = LanguageEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            output_dim=self.sensory_dim,
            config=config.spiking,
        )
        self.motor_stream = LanguageMotorStream(
            concept_dim=self.concept_dim,
            vocab_size=self.vocab_size,
            hidden_dim=self.motor_hidden_dim,
            config=config.spiking,
        )
        self.hierarchy = PredictiveHierarchy(
            layer_dims=self._build_hierarchy_dims(),
            config=config.predictive,
        )
        self.core = BioARNCore(config)
        self.connector = HierarchyConnector(self.hierarchy, self.core.ccc_pool, config.predictive)
        self.reward = RewardSystem(config.reward)
        self.text_decoder = ConceptToLanguage(
            concept_dim=self.concept_dim,
            vocab_size=self.vocab_size,
            config=config.spiking,
        )

        self.timestep = 0
        self._base_predictive_gamma = float(config.predictive.gamma)
        self._base_predictive_eta = float(config.predictive.eta)
        self._base_ccc_slow_lr = float(config.ccc.slow_lr)
        self._base_ccc_feedback_lr = float(config.ccc.feedback_lr)
        self._base_margin = float(config.margin_gate.theta_margin)

        self._last_sensory: SensoryOutput | None = None
        self._last_prediction: PredictionOutput | None = None
        self._last_perception: PerceptionOutput | None = None
        self._last_recognition: RecognitionOutput | None = None
        self._last_attention: AttentionOutput | None = None
        self._last_plan: PlanOutput | None = None
        self._last_action: ActionOutput | None = None
        self._last_reward: RewardStepOutput | None = None
        self._last_action_signal: ActionSignal | None = None
        self._feedback_features = torch.zeros(self.sensory_dim, dtype=torch.float32)
        self._generated_token_history: list[int] = []

        self._apply_modulation(self.reward.get_modulation())

    @staticmethod
    def _infer_visual_shape(input_dim: int) -> tuple[int, int, int]:
        if input_dim <= 0:
            raise ValueError("config.ccc.input_dim must be positive.")
        if input_dim % 3 == 0:
            side = int(round(math.sqrt(input_dim // 3)))
            if 3 * side * side == input_dim:
                return (3, side, side)
        side = int(round(math.sqrt(input_dim)))
        if side * side == input_dim:
            return (1, side, side)
        return (1, 1, input_dim)

    def _build_hierarchy_dims(self) -> list[int]:
        num_levels = max(2, int(self.config.predictive.num_levels))
        return [self.sensory_dim] + [self.concept_dim for _ in range(num_levels - 1)]

    def _align_dim(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        batch, squeeze = _as_batch(tensor.to(torch.float32))
        if batch.shape[-1] > target_dim:
            batch = batch[..., :target_dim]
        elif batch.shape[-1] < target_dim:
            batch = F.pad(batch, (0, target_dim - batch.shape[-1]))
        return _restore_shape(batch, squeeze)

    def _fuse_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        batched: list[torch.Tensor] = []
        squeeze = True
        batch_size = 1

        for feature in features:
            aligned = self._align_dim(feature, self.sensory_dim)
            batch, feature_squeeze = _as_batch(aligned)
            batched.append(batch)
            squeeze = squeeze and feature_squeeze
            batch_size = max(batch_size, batch.shape[0])

        stacked = torch.stack([_match_batch(batch, batch_size) for batch in batched], dim=0)
        fused = stacked.mean(dim=0)
        return _restore_shape(fused, squeeze and batch_size == 1)

    def _prepare_language_tokens(self, language_input: torch.Tensor) -> torch.Tensor:
        tokens = language_input
        if tokens.dim() == 0:
            tokens = tokens.unsqueeze(0)
        return tokens.long().remainder(self.vocab_size)

    def _dominant_concept(self) -> torch.Tensor:
        if self.core.gnw.slots:
            return self._align_dim(self.core.gnw.slots[0].direction, self.concept_dim)
        if self._last_recognition is not None:
            return self._align_dim(self._last_recognition.concept_direction, self.concept_dim)
        return torch.zeros(self.concept_dim, dtype=torch.float32)

    def _imagined_sensory(self) -> torch.Tensor:
        if torch.count_nonzero(self._feedback_features).item() > 0:
            return self._feedback_features.detach().clone()
        concept = self._dominant_concept()
        if torch.count_nonzero(concept).item() > 0:
            return self._align_dim(self.connector.top_down(concept), self.sensory_dim)
        if self._last_prediction is not None:
            return self._align_dim(self._last_prediction.prediction, self.sensory_dim)
        return torch.zeros(self.sensory_dim, dtype=torch.float32)

    def _preview_novelty(self, surprise: float, perception: PerceptionOutput) -> NoveltySignal:
        baseline = float(self.reward.prediction_error_baseline.item())
        baseline = max(baseline, 1e-6)
        novelty_ratio = surprise / baseline
        score = max(0.0, novelty_ratio - 1.0)
        is_novel = bool(
            perception.is_novel
            or (
                surprise > 0.0
                and novelty_ratio > float(self.config.reward.novelty_threshold)
            )
        )
        disruption = 1.0 if is_novel else min(1.0, score / max(self.config.reward.novelty_threshold, 1.0))
        learning_boost = 1.0 + (
            max(0.0, float(self.config.reward.novelty_boost) - 1.0) if is_novel else 0.0
        )
        return NoveltySignal(
            is_novel=is_novel,
            novelty_score=float(score),
            orienting_response=is_novel,
            learning_boost=float(learning_boost),
            attention_disruption=float(disruption),
        )

    def _learning_occurred(self, perception: PerceptionOutput) -> bool:
        if perception.pool_output.recruited:
            return True
        return any(
            output.resonance is not None
            and bool(output.resonance.resonated.reshape(-1).any().item())
            for output in perception.pool_output.outputs
            if output.fired
        )

    def _apply_modulation(self, modulation: ModulationOutput) -> None:
        learning_multiplier = max(0.1, float(modulation.learning_rate_multiplier))
        predictive_gamma = self._base_predictive_gamma * learning_multiplier
        predictive_eta = self._base_predictive_eta * learning_multiplier
        ccc_slow_lr = self._base_ccc_slow_lr * learning_multiplier
        ccc_feedback_lr = self._base_ccc_feedback_lr * learning_multiplier
        margin = min(0.95, max(0.1, self._base_margin + float(modulation.margin_adjustment)))

        self.hierarchy.config.gamma = predictive_gamma
        self.hierarchy.config.eta = predictive_eta
        for layer in self.hierarchy.layers:
            layer.config.gamma = predictive_gamma
            layer.config.eta = predictive_eta

        self.core.config.ccc.slow_lr = ccc_slow_lr
        self.core.config.ccc.feedback_lr = ccc_feedback_lr
        for ccc in self.core.ccc_pool.cccs:
            ccc.config.slow_lr = ccc_slow_lr
            ccc.config.feedback_lr = ccc_feedback_lr
            ccc.margin_gate.theta_margin.fill_(margin)

    def _compose_attention_output(
        self,
        perception: PerceptionOutput,
        novelty: NoveltySignal,
        modulation: ModulationOutput,
        shifted: bool,
    ) -> AttentionOutput:
        return AttentionOutput(
            broadcast=perception.broadcast,
            novelty=novelty,
            modulation=modulation,
            shifted=shifted,
        )

    def _effective_modulation(self, reward_step: RewardStepOutput) -> ModulationOutput:
        modulation = reward_step.modulation
        if not reward_step.novelty.is_novel:
            return modulation
        return ModulationOutput(
            learning_rate_multiplier=max(
                float(modulation.learning_rate_multiplier),
                float(reward_step.novelty.learning_boost),
            ),
            attention_disruption=max(
                float(modulation.attention_disruption),
                float(reward_step.novelty.attention_disruption),
            ),
            margin_adjustment=float(modulation.margin_adjustment),
            exploration_drive=max(
                float(modulation.exploration_drive),
                float(reward_step.novelty.attention_disruption),
            ),
        )

    def _token_from_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        ranking = torch.argsort(probabilities, dim=-1, descending=True)
        tokens = ranking[:, 0].clone()
        if self.vocab_size <= 1:
            return tokens
        eos_mask = tokens.eq(self.motor_stream.eos_token_id)
        if eos_mask.any():
            alternate = ranking[eos_mask, 1]
            tokens[eos_mask] = alternate
        return tokens

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        decoded = self.text_decoder._decode(torch.tensor(token_ids, dtype=torch.long))  # noqa: SLF001
        return decoded if decoded else "".join("?" for _ in token_ids)

    @torch.no_grad()
    def sense(
        self,
        visual_input: torch.Tensor | None = None,
        language_input: torch.Tensor | None = None,
    ) -> SensoryOutput:
        visual_output = self.visual_encoder(visual_input) if visual_input is not None else None
        language_output = (
            self.language_encoder(self._prepare_language_tokens(language_input))
            if language_input is not None
            else None
        )

        modality_features: list[torch.Tensor] = []
        suppressed: list[float] = []
        if visual_output is not None:
            modality_features.append(visual_output.features)
            suppressed.append(float(visual_output.suppressed_fraction))
        if language_output is not None:
            modality_features.append(language_output.features)
            suppressed.append(float(language_output.suppressed_fraction))

        if modality_features:
            features = self._fuse_features(modality_features)
            suppressed_fraction = float(sum(suppressed) / len(suppressed))
        else:
            features = self._imagined_sensory()
            suppressed_fraction = 0.0

        sensory = SensoryOutput(
            features=self._align_dim(features, self.sensory_dim),
            visual_output=visual_output,
            language_output=language_output,
            suppressed_fraction=suppressed_fraction,
        )
        self._last_sensory = sensory
        return sensory

    @torch.no_grad()
    def predict(self, sensory_features: torch.Tensor) -> PredictionOutput:
        aligned = self._align_dim(sensory_features, self.sensory_dim)
        compare = self.hierarchy.predict_and_compare(aligned)
        perception = self.hierarchy.perceive(aligned, num_iterations=max(4, self.config.predictive.num_levels * 2))

        concept_level = perception.states[min(self.connector.level2_index, len(perception.states) - 1)]
        generated = self.connector.top_down(concept_level)
        free_energy = (
            float(perception.free_energy_trace[-1])
            if perception.free_energy_trace
            else float(compare.error.abs().mean().item())
        )

        prediction = PredictionOutput(
            prediction=self._align_dim(generated, self.sensory_dim),
            error=self._align_dim(compare.error, self.sensory_dim),
            surprise=float(max(compare.surprise_score, perception.surprise)),
            free_energy=free_energy,
        )
        self._last_prediction = prediction
        return prediction

    @torch.no_grad()
    def recognize(self, sensory_features: torch.Tensor) -> RecognitionOutput:
        perception = self.core.perceive(self._align_dim(sensory_features, self.sensory_dim))
        abstained = perception.num_fired == 0 or perception.vote_result.voter_count == 0
        concept_direction = self._align_dim(perception.vote_result.winning_direction, self.concept_dim)
        if abstained:
            concept_direction = torch.zeros_like(concept_direction)

        recognition = RecognitionOutput(
            concept_direction=concept_direction.detach().clone(),
            confidence=float(perception.vote_result.confidence),
            abstained=abstained,
            num_hypotheses=perception.num_fired,
            agreement=float(perception.vote_result.agreement_score),
        )
        self._last_perception = perception
        self._last_recognition = recognition
        return recognition

    @torch.no_grad()
    def attend(self, perception: PerceptionOutput) -> AttentionOutput:
        if self._last_prediction is None:
            surprise = 0.0
        else:
            surprise = float(self._last_prediction.surprise)
        novelty = self._preview_novelty(surprise, perception)
        modulation = self.reward.get_modulation()
        shifted = bool(
            novelty.is_novel
            or perception.is_novel
            or bool(self.core.gnw.last_new_entries)
            or bool(perception.associations.indices)
        )
        attention = self._compose_attention_output(
            perception=perception,
            novelty=novelty,
            modulation=modulation,
            shifted=shifted,
        )
        self._last_attention = attention
        return attention

    def _compute_action_signal(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> ActionSignal:
        current_batch, current_squeeze = _as_batch(current_state.to(torch.float32))
        goal_batch, _ = _as_batch(goal_state.to(torch.float32))
        current = _restore_shape(current_batch, current_squeeze)
        goal = self._align_dim(goal_batch, current_batch.shape[-1])
        signal = self.hierarchy.active_inference_step(current, goal)
        self._last_action_signal = signal
        return signal

    @torch.no_grad()
    def plan(self, concept: torch.Tensor, goal: torch.Tensor | None = None) -> PlanOutput:
        concept_direction = self._align_dim(concept, self.concept_dim)
        motor_plan = self.motor_stream.plan(concept_direction)
        action_signal: ActionSignal | None = None

        if goal is not None:
            action_signal = self._compute_action_signal(concept_direction, goal)
            plan_batch, squeeze = _as_batch(motor_plan)
            signal_bias, _ = _as_batch(self._align_dim(action_signal.direction, plan_batch.shape[-1]))
            motor_plan = _restore_shape(
                torch.tanh(plan_batch + (0.25 * _match_batch(signal_bias, plan_batch.shape[0]))),
                squeeze,
            )

        confidence = float(torch.sigmoid(motor_plan.abs().mean()).item())
        if action_signal is not None:
            confidence = float(min(1.0, max(confidence, action_signal.expected_reduction)))

        plan_output = PlanOutput(
            motor_plan=motor_plan.detach().clone(),
            action_signal=action_signal,
            confidence=confidence,
        )
        self._last_plan = plan_output
        return plan_output

    @torch.no_grad()
    def act(self, plan: PlanOutput) -> ActionOutput:
        motor_plan, squeeze = _as_batch(plan.motor_plan.to(torch.float32))
        if self.motor_stream.prediction_buffer.numel() > 0:
            predicted, _ = _as_batch(self.motor_stream.prediction_buffer.to(motor_plan))
            predicted = _match_batch(predicted, motor_plan.shape[0])
        else:
            predicted = _as_batch(
                self.motor_stream.predict_next(
                    torch.zeros(
                        motor_plan.shape[0],
                        self.vocab_size,
                        device=motor_plan.device,
                        dtype=motor_plan.dtype,
                    ),
                    motor_plan,
                )
            )[0]

        working_plan = motor_plan
        corrections = 0
        final_step = None
        final_probabilities = None
        monitor_output = None

        for attempt in range(self.motor_stream.max_corrections_per_step + 1):
            step_output = self.motor_stream.execute_step(working_plan)
            probabilities = F.softmax(step_output.logits, dim=-1)
            monitor = self.motor_stream.self_monitor_step(probabilities, predicted)

            final_step = step_output
            final_probabilities = probabilities
            monitor_output = monitor

            if not monitor.should_revise or attempt >= self.motor_stream.max_corrections_per_step:
                break

            correction, _ = _as_batch(monitor.correction)
            working_plan = torch.tanh(
                working_plan - (self.motor_stream.correction_gain * _match_batch(correction, working_plan.shape[0]))
            )
            corrections += 1

        if final_step is None or final_probabilities is None or monitor_output is None:
            raise RuntimeError("Motor stream did not produce an action.")

        token_ids = self._token_from_probabilities(final_probabilities).reshape(-1).to(torch.long)
        logits_sequence = final_step.logits.detach().clone().reshape(-1, self.vocab_size)
        generated = GenerationOutput(
            token_ids=token_ids.detach().clone(),
            logits_sequence=logits_sequence,
            confidences=[float(final_step.confidence) for _ in range(token_ids.numel())],
            corrections_made=corrections,
            total_prediction_error=float(monitor_output.error_magnitude),
            stopped_reason="corrected" if corrections else "step",
        )

        feedback = self.language_encoder(token_ids)
        feedback_features = self._align_dim(feedback.features, self.sensory_dim)
        expected_feedback = self._imagined_sensory()
        feedback_error = float((feedback_features - self._align_dim(expected_feedback, self.sensory_dim)).abs().mean().item())
        self._feedback_features = feedback_features.detach().clone()
        self._generated_token_history.extend(token_ids.tolist())
        self.motor_stream.predict_next(final_probabilities, working_plan)

        action_output = ActionOutput(
            generated=generated,
            action_vector=_restore_shape(final_probabilities.detach().clone(), squeeze),
            self_correction=bool(corrections > 0 or feedback_error > self.motor_stream.monitor_revision_threshold),
            feedback_error=feedback_error,
        )
        self._last_action = action_output
        return action_output

    @torch.no_grad()
    def step(
        self,
        visual_input: torch.Tensor | None = None,
        language_input: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> LoopStepOutput:
        self._apply_modulation(self.reward.get_modulation())

        step_index = self.timestep
        sensory = self.sense(visual_input=visual_input, language_input=language_input)
        prediction = self.predict(sensory.features)
        recognition = self.recognize(sensory.features)

        if self._last_perception is None:
            raise RuntimeError("Recognition did not produce a perception state.")
        attention = self.attend(self._last_perception)

        concept = recognition.concept_direction
        if recognition.abstained and attention.broadcast.directions:
            concept = self._align_dim(attention.broadcast.directions[0], self.concept_dim)
        if torch.count_nonzero(concept).item() == 0:
            concept = self._dominant_concept()

        plan = self.plan(concept, goal=goal)
        action = self.act(plan)

        learned = self._learning_occurred(self._last_perception)
        self.core.learn_from_perception(self._last_perception, self._align_dim(sensory.features, self.sensory_dim))

        error_signal = float(prediction.surprise + action.feedback_error)
        reward_step = self.reward.step(error_signal, learned=learned)
        effective_modulation = self._effective_modulation(reward_step)
        reward_step = RewardStepOutput(
            reward=reward_step.reward,
            novelty=reward_step.novelty,
            modulation=effective_modulation,
            cumulative_reward=reward_step.cumulative_reward,
            steps_since_novelty=reward_step.steps_since_novelty,
        )
        self._apply_modulation(reward_step.modulation)

        pool_stats = self.core.ccc_pool.get_pool_stats()
        for ccc in self.core.ccc_pool.cccs:
            if bool(ccc.is_committed.item()):
                ccc.margin_gate.adapt_threshold(float(pool_stats["fire_rate"]))

        attention = self._compose_attention_output(
            perception=self._last_perception,
            novelty=reward_step.novelty,
            modulation=reward_step.modulation,
            shifted=bool(attention.shifted or reward_step.novelty.is_novel),
        )

        output = LoopStepOutput(
            sensory=sensory,
            prediction=prediction,
            recognition=recognition,
            attention=attention,
            plan=plan,
            action=action,
            reward=reward_step,
            learned=learned,
            timestep=step_index,
        )
        self._last_reward = reward_step
        self.timestep += 1
        return output

    def _route_run_input(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if value.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.long,
        }:
            return None, value
        return value, None

    @torch.no_grad()
    def run(
        self,
        inputs: list[torch.Tensor],
        num_steps: int | None = None,
        generate: bool = False,
    ) -> LoopRunOutput:
        steps: list[LoopStepOutput] = []
        free_energy_trace: list[float] = []
        reward_trace: list[float] = []
        novelty_events: list[int] = []
        generated_tokens: list[int] = []

        if generate:
            total_steps = max(1, num_steps or len(inputs) or 1)
            iterable: list[torch.Tensor | None] = [None for _ in range(total_steps)]
        else:
            if not inputs:
                raise ValueError("inputs must be non-empty when generate=False.")
            total_steps = len(inputs) if num_steps is None else min(len(inputs), max(1, num_steps))
            iterable = list(inputs[:total_steps])

        for current in iterable:
            if generate:
                step_output = self.step()
            else:
                visual_input, language_input = self._route_run_input(current)
                step_output = self.step(visual_input=visual_input, language_input=language_input)

            steps.append(step_output)
            free_energy_trace.append(float(step_output.prediction.free_energy))
            reward_trace.append(float(step_output.reward.reward.value))

            if step_output.reward.novelty.is_novel:
                novelty_events.append(step_output.timestep)

            if step_output.action is not None and step_output.action.generated is not None:
                generated_tokens.extend(step_output.action.generated.token_ids.tolist())
                if generate:
                    low_confidence = (
                        step_output.action.generated.confidences
                        and step_output.action.generated.confidences[-1] < self.motor_stream.confidence_threshold
                    )
                    if low_confidence:
                        break

        generated_text = self._decode_tokens(generated_tokens) if generate else None
        total_learning_events = sum(1 for step_output in steps if step_output.learned)
        reward_stats = self.reward.get_stats()
        final_stats = {
            "steps": len(steps),
            "final_free_energy": free_energy_trace[-1] if free_energy_trace else 0.0,
            "mean_free_energy": sum(free_energy_trace) / len(free_energy_trace) if free_energy_trace else 0.0,
            "mean_reward": sum(reward_trace) / len(reward_trace) if reward_trace else 0.0,
            "cumulative_reward": float(reward_stats["cumulative_reward"]),
            "novelty_events": len(novelty_events),
            "concepts_learned": int(self.core.get_system_stats()["concepts_learned"]),
            "workspace_occupancy": float(self.core.gnw.get_stats()["occupancy"]),
            "generated_tokens": len(generated_tokens),
            "last_modulation": reward_stats["modulation"],
        }
        return LoopRunOutput(
            steps=steps,
            free_energy_trace=free_energy_trace,
            reward_trace=reward_trace,
            novelty_events=novelty_events,
            generated_text=generated_text,
            total_learning_events=total_learning_events,
            final_stats=final_stats,
        )

    @torch.no_grad()
    def generate_text(self, seed_concept: torch.Tensor, max_tokens: int = 50) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        concept = self._align_dim(seed_concept, self.concept_dim)
        self.core.gnw.inject(0, concept, priority=1.0)
        output = self.run(inputs=[], num_steps=max_tokens, generate=True)
        if output.generated_text:
            return output.generated_text

        fallback = self.text_decoder.narrate(concept, max_words=max(1, max_tokens // 4))
        return fallback if fallback else self._decode_tokens(self._generated_token_history[-max_tokens:])

    @torch.no_grad()
    def active_inference_step(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        signal = self._compute_action_signal(current_state, goal_state)
        return signal.direction.detach().clone()


__all__ = [
    "ActionOutput",
    "AttentionOutput",
    "LoopRunOutput",
    "LoopStepOutput",
    "PlanOutput",
    "PredictionOutput",
    "SensorimotorLoop",
    "SensoryOutput",
]
