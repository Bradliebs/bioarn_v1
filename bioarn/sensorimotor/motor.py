"""Motor cortex streams for predictive language and action generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import PredictiveConfig, SpikingConfig
from bioarn.core.math_utils import normalize
from bioarn.core.spiking import LIFLayer


@dataclass
class MotorStepOutput:
    """Outputs from one motor execution step."""

    logits: torch.Tensor
    confidence: float
    spike_state: torch.Tensor
    is_eos: bool


@dataclass
class GenerationOutput:
    """Outputs from a full language generation episode."""

    token_ids: torch.Tensor
    logits_sequence: torch.Tensor
    confidences: list[float]
    corrections_made: int
    total_prediction_error: float
    stopped_reason: str


@dataclass
class MonitorOutput:
    """Outputs from self-monitoring and self-correction."""

    error: torch.Tensor
    error_magnitude: float
    correction: torch.Tensor
    should_revise: bool


@dataclass
class ActionOutput:
    """Outputs from physical motor execution."""

    action_vector: torch.Tensor
    confidence: float


def _fill_deterministic_linear(linear: nn.Linear, *, gain: float, phase: float) -> None:
    with torch.no_grad():
        out_axis = torch.arange(
            linear.out_features,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        ).unsqueeze(1)
        in_axis = torch.arange(
            linear.in_features,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        ).unsqueeze(0)
        weights = (
            torch.sin((out_axis + 1.0) * (in_axis + 1.0) * phase)
            + torch.cos((out_axis + 1.0) * (in_axis + 1.0) * (phase * 0.61))
        ) * 0.5
        linear.weight.copy_(normalize(weights) * gain)
        if linear.bias is not None:
            linear.bias.zero_()


class LanguageMotorStream(nn.Module):
    """Predictive motor stream that turns concepts into language output."""

    def __init__(
        self,
        concept_dim: int,
        vocab_size: int,
        hidden_dim: int = 128,
        config: SpikingConfig | None = None,
    ) -> None:
        super().__init__()
        if concept_dim <= 0 or vocab_size <= 0 or hidden_dim <= 0:
            raise ValueError("concept_dim, vocab_size, and hidden_dim must be positive.")

        self.concept_dim = int(concept_dim)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.config = config or SpikingConfig()
        self.predictive_config = PredictiveConfig()
        self.eos_token_id = self.vocab_size - 1
        self.max_corrections_per_step = 3
        self.confidence_threshold = min(0.35, max(0.10, 2.5 / self.vocab_size))
        self.eos_confidence_threshold = max(0.45, self.confidence_threshold + 0.15)
        self.monitor_revision_threshold = max(0.05, self.predictive_config.error_threshold * 5.0)
        self.correction_gain = 0.5

        self.concept_to_motor = nn.Linear(self.concept_dim, self.hidden_dim)
        self.motor_planner = LIFLayer(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            config=self.config,
            spike_history_steps=8,
        )
        self.motor_executor = LIFLayer(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            config=self.config,
            spike_history_steps=8,
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        self.self_monitor = nn.Linear(self.vocab_size, self.hidden_dim, bias=False)
        self.predictive_model = nn.Linear(self.hidden_dim + self.vocab_size, self.vocab_size, bias=False)

        self.register_buffer("prediction_buffer", torch.empty(0))
        self.register_buffer("last_motor_state", torch.empty(0))
        self.register_buffer("last_output_distribution", torch.empty(0))

        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self) -> None:
        _fill_deterministic_linear(self.concept_to_motor, gain=1.5, phase=0.071)
        _fill_deterministic_linear(self.motor_planner.linear, gain=1.9, phase=0.049)
        _fill_deterministic_linear(self.motor_executor.linear, gain=1.7, phase=0.063)
        _fill_deterministic_linear(self.output_projection, gain=1.35, phase=0.091)
        _fill_deterministic_linear(self.self_monitor, gain=0.8, phase=0.077)
        _fill_deterministic_linear(self.predictive_model, gain=1.1, phase=0.057)

    @staticmethod
    def _batchify(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        if tensor.dim() == 2:
            return tensor, False
        raise ValueError("Expected a 1D or 2D tensor.")

    def plan(self, concept_direction: torch.Tensor) -> torch.Tensor:
        """Convert a concept direction into a hidden motor plan."""

        concept_batch, _ = self._batchify(concept_direction.to(torch.float32))
        projected = torch.tanh(self.concept_to_motor(concept_batch))
        plan_spikes, plan_membrane = self.motor_planner(projected)
        motor_plan = torch.tanh(projected + plan_membrane + (0.5 * plan_spikes))
        self.last_motor_state = motor_plan.detach().clone()
        return motor_plan

    def execute_step(self, motor_plan: torch.Tensor) -> MotorStepOutput:
        """Produce one predictive output step from the current motor plan."""

        plan_batch, _ = self._batchify(motor_plan.to(torch.float32))
        exec_spikes, exec_membrane = self.motor_executor(plan_batch)
        spike_state = torch.tanh(plan_batch + exec_membrane + exec_spikes)
        logits = self.output_projection(spike_state)
        probabilities = F.softmax(logits, dim=-1)
        confidence = float(probabilities.max(dim=-1).values.mean().item())
        top_token = int(probabilities.argmax(dim=-1)[0].item())
        is_eos = top_token == self.eos_token_id and confidence >= self.eos_confidence_threshold

        self.last_motor_state = spike_state.detach().clone()
        self.last_output_distribution = probabilities.detach().clone()
        return MotorStepOutput(
            logits=logits,
            confidence=confidence,
            spike_state=spike_state,
            is_eos=is_eos,
        )

    def self_monitor_step(self, produced: torch.Tensor, predicted: torch.Tensor) -> MonitorOutput:
        """Compare produced output against the current top-down prediction."""

        produced_batch, produced_squeezed = self._batchify(produced.to(torch.float32))
        predicted_batch, _ = self._batchify(predicted.to(torch.float32))
        if produced_batch.shape != predicted_batch.shape:
            raise ValueError("produced and predicted must have identical shapes.")

        error = produced_batch - predicted_batch
        error_magnitude = float(error.abs().mean().item())
        correction_strength = error.abs().mean(dim=-1, keepdim=True)
        correction = torch.tanh(self.self_monitor(error)) * correction_strength
        should_revise = error_magnitude > self.monitor_revision_threshold

        if produced_squeezed:
            error = error.squeeze(0)
            correction = correction.squeeze(0)

        return MonitorOutput(
            error=error,
            error_magnitude=error_magnitude,
            correction=correction,
            should_revise=should_revise,
        )

    def predict_next(self, current_output: torch.Tensor, motor_state: torch.Tensor) -> torch.Tensor:
        """Predict the next output distribution from current output and motor state."""

        output_batch, output_squeezed = self._batchify(current_output.to(torch.float32))
        state_batch, _ = self._batchify(motor_state.to(torch.float32))
        if output_batch.shape[0] != state_batch.shape[0]:
            raise ValueError("current_output and motor_state batch sizes must match.")

        predictor_input = torch.cat((output_batch, state_batch), dim=-1)
        predicted_logits = torch.tanh(self.predictive_model(predictor_input))
        predicted = F.softmax(predicted_logits, dim=-1)
        self.prediction_buffer = predicted.detach().clone()
        return predicted.squeeze(0) if output_squeezed else predicted

    def generate_sequence(
        self,
        concept_direction: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> GenerationOutput:
        """Generate a language sequence through predictive execution and monitoring."""

        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        concept_batch, _ = self._batchify(concept_direction.to(torch.float32))
        if concept_batch.shape[0] != 1:
            raise ValueError("generate_sequence currently supports one concept at a time.")

        self.reset()
        motor_plan = self.plan(concept_batch)
        zero_output = torch.zeros(1, self.vocab_size, device=motor_plan.device, dtype=motor_plan.dtype)
        predicted = self.predict_next(zero_output, motor_plan)

        token_ids: list[int] = []
        logits_sequence: list[torch.Tensor] = []
        confidences: list[float] = []
        corrections_made = 0
        total_prediction_error = 0.0
        stopped_reason = "max_length"

        for _ in range(max_length):
            working_plan = motor_plan
            final_step: MotorStepOutput | None = None
            final_probs: torch.Tensor | None = None

            for correction_attempt in range(self.max_corrections_per_step + 1):
                step_output = self.execute_step(working_plan)
                scaled_logits = step_output.logits / float(temperature)
                probabilities = F.softmax(scaled_logits, dim=-1)
                confidence = float(probabilities.max(dim=-1).values.item())
                top_token = int(probabilities.argmax(dim=-1).item())
                is_eos = top_token == self.eos_token_id and confidence >= self.eos_confidence_threshold
                monitor = self.self_monitor_step(probabilities, predicted)
                total_prediction_error += monitor.error_magnitude

                final_step = MotorStepOutput(
                    logits=scaled_logits.detach().clone(),
                    confidence=confidence,
                    spike_state=step_output.spike_state,
                    is_eos=is_eos,
                )
                final_probs = probabilities.detach().clone()

                if not monitor.should_revise or correction_attempt >= self.max_corrections_per_step:
                    break

                working_plan = torch.tanh(working_plan - (self.correction_gain * monitor.correction))
                corrections_made += 1

            if final_step is None or final_probs is None:
                raise RuntimeError("Generation step did not produce output.")

            token_id = int(final_probs.argmax(dim=-1).item())
            token_ids.append(token_id)
            logits_sequence.append(final_step.logits.squeeze(0))
            confidences.append(final_step.confidence)

            motor_plan = torch.tanh(working_plan + (0.25 * final_step.spike_state))
            self.last_motor_state = motor_plan.detach().clone()
            predicted = self.predict_next(final_probs, motor_plan)

            if final_step.is_eos:
                stopped_reason = "eos"
                break
            if final_step.confidence < self.confidence_threshold:
                stopped_reason = "low_confidence"
                break
        else:
            stopped_reason = "max_length"

        token_tensor = torch.tensor(token_ids, device=motor_plan.device, dtype=torch.long)
        logits_tensor = (
            torch.stack(logits_sequence, dim=0)
            if logits_sequence
            else torch.empty(0, self.vocab_size, device=motor_plan.device, dtype=motor_plan.dtype)
        )
        return GenerationOutput(
            token_ids=token_tensor,
            logits_sequence=logits_tensor,
            confidences=confidences,
            corrections_made=corrections_made,
            total_prediction_error=float(total_prediction_error),
            stopped_reason=stopped_reason,
        )

    def reset(self) -> None:
        """Clear internal state for a fresh generation episode."""

        self.motor_planner.reset_state()
        self.motor_executor.reset_state()
        device = self.concept_to_motor.weight.device
        dtype = self.concept_to_motor.weight.dtype
        self.prediction_buffer = torch.empty(0, device=device, dtype=dtype)
        self.last_motor_state = torch.empty(0, device=device, dtype=dtype)
        self.last_output_distribution = torch.empty(0, device=device, dtype=dtype)


class PhysicalMotorStream(nn.Module):
    """Placeholder physical-action motor stream for later embodiment phases."""

    def __init__(self, concept_dim: int, action_dim: int, config: SpikingConfig | None = None) -> None:
        super().__init__()
        if concept_dim <= 0 or action_dim <= 0:
            raise ValueError("concept_dim and action_dim must be positive.")

        self.concept_dim = int(concept_dim)
        self.action_dim = int(action_dim)
        self.config = config or SpikingConfig()

        self.concept_to_action = nn.Linear(self.concept_dim, self.action_dim)
        self.action_planner = LIFLayer(
            self.action_dim,
            self.action_dim,
            bias=False,
            config=self.config,
            spike_history_steps=4,
        )
        self.action_projection = nn.Linear(self.action_dim, self.action_dim)

        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self) -> None:
        _fill_deterministic_linear(self.concept_to_action, gain=1.35, phase=0.083)
        _fill_deterministic_linear(self.action_planner.linear, gain=1.5, phase=0.069)
        _fill_deterministic_linear(self.action_projection, gain=1.0, phase=0.097)

    def plan_action(self, concept: torch.Tensor) -> torch.Tensor:
        """Convert a concept direction into an embodied action plan."""

        concept_batch, _ = LanguageMotorStream._batchify(concept.to(torch.float32))
        projected = torch.tanh(self.concept_to_action(concept_batch))
        spikes, membrane = self.action_planner(projected)
        return torch.tanh(projected + membrane + (0.5 * spikes))

    def execute_action(self, plan: torch.Tensor) -> ActionOutput:
        """Convert an action plan into primitive actuator values."""

        plan_batch, squeezed = LanguageMotorStream._batchify(plan.to(torch.float32))
        action_vector = torch.tanh(self.action_projection(plan_batch))
        confidence = float(action_vector.abs().mean().item())
        if squeezed:
            action_vector = action_vector.squeeze(0)
        return ActionOutput(action_vector=action_vector, confidence=confidence)

    def reset(self) -> None:
        """Clear spiking state for a new action episode."""

        self.action_planner.reset_state()


class ConceptToLanguage(nn.Module):
    """High-level wrapper from GNW concept directions to decoded text."""

    def __init__(self, concept_dim: int, vocab_size: int, config: SpikingConfig | None = None) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.motor_stream = LanguageMotorStream(
            concept_dim=concept_dim,
            vocab_size=vocab_size,
            hidden_dim=max(64, min(128, vocab_size * 4)),
            config=config,
        )
        self.default_vocab = self._build_default_vocab(self.vocab_size)

    @staticmethod
    def _build_default_vocab(vocab_size: int) -> dict[int, str]:
        symbols = list(" etaoinshrdlucmfwypvbgkjqxz0123456789,.;:!?-")
        vocab: dict[int, str] = {}
        for idx in range(vocab_size):
            if idx == vocab_size - 1:
                vocab[idx] = ""
            elif idx < len(symbols):
                vocab[idx] = symbols[idx]
            else:
                vocab[idx] = chr(97 + ((idx - 1) % 26))
        return vocab

    @staticmethod
    def _invert_vocab(vocab: dict) -> dict[int, str]:
        if not vocab:
            return {}
        first_key = next(iter(vocab))
        if isinstance(first_key, int):
            return {int(key): str(value) for key, value in vocab.items()}
        return {int(value): str(key) for key, value in vocab.items()}

    def _decode(self, token_ids: torch.Tensor, vocab: dict[int, str] | None = None) -> str:
        id_to_token = vocab or self.default_vocab
        decoded = []
        for token_id in token_ids.tolist():
            token = id_to_token.get(int(token_id), "")
            if token.lower() in {"<eos>", "[eos]"}:
                continue
            decoded.append(token)
        return "".join(decoded).strip()

    def speak(self, thought_chain: list[torch.Tensor], vocab: dict) -> str:
        """Generate language for each concept in a thought chain and concatenate it."""

        id_to_token = self._invert_vocab(vocab)
        segments = []
        for thought in thought_chain:
            generated = self.motor_stream.generate_sequence(
                thought,
                max_length=max(4, min(12, self.vocab_size)),
            )
            segment = self._decode(generated.token_ids, id_to_token)
            if segment:
                segments.append(segment)
        return " ".join(segments).strip()

    def narrate(self, concept_direction: torch.Tensor, max_words: int = 20) -> str:
        """Generate a free-form narration from a single concept direction."""

        max_length = max(1, max_words * 4)
        generated = self.motor_stream.generate_sequence(concept_direction, max_length=max_length)
        return self._decode(generated.token_ids)


__all__ = [
    "ActionOutput",
    "ConceptToLanguage",
    "GenerationOutput",
    "LanguageMotorStream",
    "MonitorOutput",
    "MotorStepOutput",
    "PhysicalMotorStream",
]
