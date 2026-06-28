"""Bio-ARN world model for curiosity-driven reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from bioarn.config import (
    CCCConfig,
    LateralPredictionConfig,
    MarginGateConfig,
    PrecisionConfig,
    RewardConfig,
    WorldModelConfig,
)
from bioarn.core.ccc import CCCPool, CCCPoolOutput
from bioarn.core.math_utils import normalize
from bioarn.reward import RewardSystem


@dataclass
class StateRepresentation:
    """Sparse concept-based representation of an environment observation."""

    observation: torch.Tensor
    concept_vector: torch.Tensor
    fired_indices: list[int]
    winner_confidences: torch.Tensor
    concept_id: int | None
    precision: float
    uncertainty: float
    novelty: float
    predicted_reward: float = 0.0
    prediction_confidence: float = 0.0
    metadata: dict[str, float | int | bool] = field(default_factory=dict)


class BioARNWorldModel:
    """Bio-ARN as a world model for RL.

    CCCs discretize observations into reusable concept states. A precision signal
    converts state uncertainty into curiosity, and action-conditioned Hebbian
    transition links provide next-state predictions without backpropagation.
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config
        num_f1_features = max(self.config.concept_dim, self.config.observation_dim * 8)
        precision_config = (
            PrecisionConfig(
                enabled=True,
                pool_size=self.config.max_pool_size,
                entropy_window=max(25, self.config.max_pool_size),
                precision_alpha=5.0,
                precision_threshold=0.4,
                min_precision=0.1,
                max_precision=1.0,
                lateral_error_weight=0.45,
                hierarchy_error_weight=0.15,
                external_signal_decay=0.9,
                surprise_gain=1.5,
            )
            if self.config.use_precision
            else None
        )
        lateral_config = LateralPredictionConfig(
            enabled=True,
            max_neighbors=min(8, max(1, self.config.max_pool_size - 1)),
            hebbian_lr=self.config.prediction_lr,
            anti_hebbian_lr=self.config.prediction_lr * 0.4,
            min_weight=0.05,
            max_weight=2.5,
            refresh_interval=8,
            prediction_threshold=0.1,
            surprise_gain=1.5,
        )
        pool_config = CCCConfig(
            input_dim=self.config.observation_dim,
            concept_dim=self.config.concept_dim,
            num_f1_features=num_f1_features,
            f1_top_k=min(num_f1_features, max(4, self.config.observation_dim * 2)),
            fast_lr=1.0,
            slow_lr=0.2,
            feedback_lr=0.12,
            max_pool_size=self.config.max_pool_size,
            max_growth_factor=1.0,
            consolidation_strength=0.0,
            lock_threshold=1.1,
            precision=precision_config,
            lateral_prediction=lateral_config,
        )
        self.pool = CCCPool(
            pool_config,
            MarginGateConfig(theta_margin=0.35, theta_margin_lr=0.001, theta_resonance=0.8),
        )
        self.reward_system = RewardSystem(
            RewardConfig(
                intrinsic_scale=1.0,
                novelty_threshold=1.25,
                novelty_boost=1.75,
                novelty_decay=0.95,
                curiosity_weight=self.config.curiosity_weight,
            )
        )
        self.precision_gate = self.pool.precision_gate
        self.transition_model = torch.zeros(
            self.config.num_actions,
            self.config.max_pool_size,
            self.config.max_pool_size,
            dtype=torch.float32,
        )
        self.transition_counts = torch.zeros_like(self.transition_model)
        self.reward_model = torch.zeros(
            self.config.max_pool_size,
            self.config.num_actions,
            dtype=torch.float32,
        )
        self.state_visit_counts = torch.zeros(self.config.max_pool_size, dtype=torch.float32)
        self.observation_prototypes = torch.zeros(
            self.config.max_pool_size,
            self.config.observation_dim,
            dtype=torch.float32,
        )
        self.prototype_counts = torch.zeros(self.config.max_pool_size, dtype=torch.float32)
        self.observation_scale = torch.ones(self.config.observation_dim, dtype=torch.float32)
        self.timestep = 0
        self.last_prediction_error = 0.0
        self.last_update_stats: dict[str, float | int | bool] = {}
        self._initialize_encoder()

    def _initialize_encoder(self) -> None:
        shared = self.pool.cccs[0].f1_layer
        with torch.no_grad():
            weights = torch.zeros_like(shared.weight)
            eye = torch.eye(self.config.observation_dim, dtype=weights.dtype)
            row = 0
            for block in (eye, -eye, 0.5 * eye, -0.5 * eye):
                take = min(block.shape[0], weights.shape[0] - row)
                if take <= 0:
                    break
                weights[row : row + take].copy_(block[:take])
                row += take
            if row < weights.shape[0]:
                random_block = torch.randn(
                    weights.shape[0] - row,
                    weights.shape[1],
                    generator=torch.Generator().manual_seed(7),
                    dtype=weights.dtype,
                )
                weights[row:].copy_(normalize(random_block))
            shared.weight.copy_(weights)
            shared.bias.zero_()

    def set_observation_scale(self, scale: torch.Tensor | list[float] | tuple[float, ...]) -> None:
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if scale_tensor.numel() != self.config.observation_dim:
            raise ValueError(
                f"Expected observation scale of length {self.config.observation_dim}, got {scale_tensor.numel()}."
            )
        self.observation_scale = scale_tensor.clamp_min(1e-3)

    def reset_episode(self) -> None:
        """Reset only transient modulatory state between episodes."""

        self.reward_system.reset()
        self.last_prediction_error = 0.0
        self.last_update_stats = {}

    def _preprocess_observation(
        self,
        observation: torch.Tensor,
        *,
        update_scale: bool = True,
    ) -> torch.Tensor:
        flat = observation.detach().to(dtype=torch.float32).reshape(-1)
        if flat.numel() != self.config.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.config.observation_dim}, got {flat.numel()} values."
            )
        if update_scale:
            self.observation_scale = torch.maximum(
                self.observation_scale,
                flat.abs().clamp_min(1e-3),
            )
        scaled = torch.tanh(flat / self.observation_scale.clamp_min(1e-3))
        if float(scaled.norm().item()) <= 1e-4:
            scaled = scaled.clone()
            scaled[0] = 0.5
        return scaled

    def _active_directions(self, indices: list[int], weights: torch.Tensor | None = None) -> torch.Tensor:
        if not indices:
            return torch.zeros(self.config.concept_dim, dtype=torch.float32)
        directions = torch.stack(
            [self.pool.cccs[index].concept_direction.detach().to(torch.float32) for index in indices],
            dim=0,
        )
        if weights is None or weights.numel() == 0:
            aggregate = directions.mean(dim=0)
        else:
            normalized_weights = weights.reshape(-1).to(torch.float32)
            normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-6)
            aggregate = (normalized_weights.unsqueeze(-1) * directions).sum(dim=0)
        if float(aggregate.norm().item()) <= 1e-8:
            return torch.zeros_like(aggregate)
        return F.normalize(aggregate, dim=0)

    def _build_state(
        self,
        observation: torch.Tensor,
        pool_output: CCCPoolOutput,
        *,
        update_memory: bool = True,
        prediction_confidence: float = 0.0,
        predicted_reward: float = 0.0,
        concept_vector: torch.Tensor | None = None,
        uncertainty_override: float | None = None,
    ) -> StateRepresentation:
        concept_id = None
        if pool_output.fired_indices:
            best_position = int(torch.argmax(pool_output.winner_confidences).item())
            concept_id = int(pool_output.fired_indices[best_position])
        elif pool_output.recruited_index is not None:
            concept_id = int(pool_output.recruited_index)

        if concept_vector is None:
            concept_vector = self._active_directions(
                pool_output.fired_indices,
                pool_output.winner_confidences,
            )
        precision = float(self.pool.get_precision())
        uncertainty = (
            float(self.precision_gate.current_uncertainty)
            if self.precision_gate is not None
            else 1.0
        )
        if uncertainty_override is not None:
            uncertainty = float(max(0.0, min(1.0, uncertainty_override)))
            if self.precision_gate is not None:
                precision = float(self.precision_gate.precision_signal.compute(uncertainty))

        novelty = 1.0
        if concept_id is not None and update_memory:
            self.state_visit_counts[concept_id] += 1.0
            novelty = float(1.0 / math.sqrt(1.0 + float(self.state_visit_counts[concept_id].item())))
            self._update_observation_prototype(concept_id, observation)
        elif concept_id is not None:
            novelty = float(1.0 / math.sqrt(1.0 + float(self.state_visit_counts[concept_id].item())))
        return StateRepresentation(
            observation=observation.detach().to(torch.float32).reshape(-1),
            concept_vector=concept_vector.detach().to(torch.float32),
            fired_indices=[int(index) for index in pool_output.fired_indices],
            winner_confidences=pool_output.winner_confidences.detach().to(torch.float32),
            concept_id=concept_id,
            precision=precision,
            uncertainty=uncertainty,
            novelty=novelty,
            predicted_reward=float(predicted_reward),
            prediction_confidence=float(max(0.0, min(1.0, prediction_confidence))),
            metadata={
                "recruited": bool(pool_output.recruited),
                "num_fired": len(pool_output.fired_indices),
            },
        )

    def _update_observation_prototype(self, concept_id: int, observation: torch.Tensor) -> None:
        if not 0 <= int(concept_id) < self.config.max_pool_size:
            return
        self.prototype_counts[concept_id] += 1.0
        lr = 1.0 / float(self.prototype_counts[concept_id].item())
        self.observation_prototypes[concept_id].lerp_(
            observation.detach().to(torch.float32).reshape(-1),
            lr,
        )

    def encode_state(self, observation: torch.Tensor) -> StateRepresentation:
        """Encode environment observation into a CCC state representation."""

        obs = observation.detach().to(torch.float32).reshape(-1)
        scaled = self._preprocess_observation(obs)
        modulation = self.reward_system.get_modulation()
        pool_output = self.pool(
            scaled,
            timestep=self.timestep,
            learning_rate_multiplier=modulation.learning_rate_multiplier,
        )
        self.timestep += 1
        return self._build_state(obs, pool_output)

    def _default_prediction(self, state: StateRepresentation) -> StateRepresentation:
        confidence = 0.0
        concept_vector = state.concept_vector.detach().clone()
        if self.pool.lateral_network is not None and state.fired_indices:
            lateral = self.pool.lateral_network.predict_lateral(
                state.fired_indices,
                self.pool.concept_directions,
            )
            if lateral:
                indices = list(lateral.keys())
                weights = torch.tensor(
                    [float(value.item()) for value in lateral.values()],
                    dtype=torch.float32,
                )
                concept_vector = self._active_directions(indices, weights)
                confidence = float(weights.max().item())
        observation = state.observation.detach().clone()
        pool_output = self.pool.preview(self._preprocess_observation(observation, update_scale=False))
        return self._build_state(
            observation,
            pool_output,
            update_memory=False,
            prediction_confidence=confidence,
            predicted_reward=state.predicted_reward,
            concept_vector=concept_vector,
            uncertainty_override=1.0 - confidence,
        )

    def expected_reward(self, state: StateRepresentation, action: int) -> float:
        """Estimate expected immediate reward for a state-action pair."""

        if state.concept_id is None:
            return 0.0
        return float(self.reward_model[state.concept_id, int(action)].item())

    def transition_confidence(self, state: StateRepresentation, action: int) -> float:
        """Return how confident the transition model is about this action."""

        if state.concept_id is None:
            return 0.0
        row = self.transition_model[int(action), state.concept_id]
        total = float(row.sum().item())
        if total <= 1e-6:
            return 0.0
        return float(row.max().item())

    def predict_next_state(self, state: StateRepresentation, action: int) -> StateRepresentation:
        """Predict next state given current state and action."""

        action_index = int(action)
        if not 0 <= action_index < self.config.num_actions:
            raise ValueError(f"Action {action} is out of range for {self.config.num_actions} actions.")
        if not state.fired_indices and state.concept_id is None:
            return self._default_prediction(state)

        source_indices = state.fired_indices or ([state.concept_id] if state.concept_id is not None else [])
        source_weights = (
            state.winner_confidences.detach().to(torch.float32)
            if state.winner_confidences.numel() == len(source_indices)
            else torch.ones(len(source_indices), dtype=torch.float32)
        )
        source_weights = source_weights / source_weights.sum().clamp_min(1e-6)
        target_distribution = torch.zeros(self.config.max_pool_size, dtype=torch.float32)
        predicted_reward = 0.0
        for weight, source_index in zip(source_weights.tolist(), source_indices, strict=False):
            row = self.transition_model[action_index, int(source_index)]
            target_distribution.add_(float(weight) * row)
            predicted_reward += float(weight) * self.expected_reward(state, action_index)

        total_mass = float(target_distribution.sum().item())
        if total_mass <= 1e-6:
            return self._default_prediction(state)

        target_distribution /= total_mass
        top_values, top_indices = torch.topk(
            target_distribution,
            k=min(3, self.config.max_pool_size),
        )
        active_mask = top_values > 0.05
        predicted_indices = [int(index) for index in top_indices[active_mask].tolist()]
        predicted_weights = top_values[active_mask]
        concept_vector = self._active_directions(predicted_indices, predicted_weights)
        if predicted_indices:
            obs_weights = predicted_weights / predicted_weights.sum().clamp_min(1e-6)
            observation = (
                obs_weights.unsqueeze(-1)
                * self.observation_prototypes[predicted_indices].to(torch.float32)
            ).sum(dim=0)
        else:
            observation = state.observation.detach().clone()

        entropy = 0.0
        valid = target_distribution[target_distribution > 0]
        if valid.numel() > 0:
            entropy = float((-(valid * valid.log()).sum() / math.log(max(valid.numel(), 2))).item())
        predicted_pool_output = self.pool.preview(
            self._preprocess_observation(observation, update_scale=False)
        )
        return self._build_state(
            observation,
            predicted_pool_output,
            update_memory=False,
            prediction_confidence=float(top_values.max().item()),
            predicted_reward=predicted_reward,
            concept_vector=concept_vector,
            uncertainty_override=entropy,
        )

    def compute_curiosity(self, state: StateRepresentation) -> float:
        """Intrinsic motivation from precision/uncertainty."""

        exploration_drive = float(self.reward_system.get_modulation().exploration_drive)
        uncertainty_drive = max(0.0, min(1.0, float(state.uncertainty)))
        prediction_uncertainty = 1.0 - max(0.0, min(1.0, float(state.prediction_confidence)))
        visit_uncertainty = 1.0
        prototype_mismatch = 0.0
        if state.concept_id is not None:
            visits = float(self.state_visit_counts[state.concept_id].item())
            visit_uncertainty = 1.0 / math.sqrt(1.0 + max(visits, 0.0))
            if float(self.prototype_counts[state.concept_id].item()) > 0.0:
                normalized_delta = (
                    (state.observation - self.observation_prototypes[state.concept_id])
                    / self.observation_scale.clamp_min(1e-3)
                ).abs()
                prototype_mismatch = float(normalized_delta.mean().clamp(0.0, 1.0).item())
        novelty_drive = max(0.0, min(1.0, float(state.novelty)))
        curiosity = (
            (0.15 * uncertainty_drive)
            + (0.35 * prediction_uncertainty)
            + (0.25 * max(novelty_drive, visit_uncertainty))
            + (0.15 * prototype_mismatch)
            + (0.1 * exploration_drive)
        )
        if bool(state.metadata.get("recruited", False)):
            curiosity += 0.15
        return float(max(0.0, min(1.5, curiosity * max(self.config.curiosity_weight, 1e-6))))

    def update(
        self,
        state: StateRepresentation,
        action: int,
        reward: float,
        next_state: StateRepresentation,
    ) -> dict[str, float | int | bool]:
        """Update the Hebbian transition model from one transition."""

        predicted_state = self.predict_next_state(state, action)
        if (
            float(predicted_state.concept_vector.norm().item()) > 1e-8
            and float(next_state.concept_vector.norm().item()) > 1e-8
        ):
            similarity = float(
                F.cosine_similarity(
                    predicted_state.concept_vector.unsqueeze(0),
                    next_state.concept_vector.unsqueeze(0),
                ).item()
            )
            prediction_error = max(0.0, 1.0 - similarity)
        else:
            prediction_error = 1.0 if next_state.concept_id != predicted_state.concept_id else 0.0

        self.pool.set_hierarchy_prediction_error(prediction_error)
        self.reward_system.apply_external_reward(float(reward))
        reward_step = self.reward_system.step(prediction_error, learned=next_state.concept_id is not None)
        lr = float(self.config.prediction_lr) * max(0.25, reward_step.modulation.learning_rate_multiplier)
        source_indices = state.fired_indices or ([state.concept_id] if state.concept_id is not None else [])
        target_indices = next_state.fired_indices or ([next_state.concept_id] if next_state.concept_id is not None else [])

        source_weights = (
            state.winner_confidences.detach().to(torch.float32)
            if state.winner_confidences.numel() == len(source_indices)
            else torch.ones(len(source_indices), dtype=torch.float32)
        )
        target_weights = (
            next_state.winner_confidences.detach().to(torch.float32)
            if next_state.winner_confidences.numel() == len(target_indices)
            else torch.ones(len(target_indices), dtype=torch.float32)
        )
        if source_weights.numel() > 0:
            source_weights = source_weights / source_weights.sum().clamp_min(1e-6)
        if target_weights.numel() > 0:
            target_weights = target_weights / target_weights.sum().clamp_min(1e-6)

        action_index = int(action)
        for source_strength, source_index in zip(source_weights.tolist(), source_indices, strict=False):
            row = self.transition_model[action_index, int(source_index)]
            count_row = self.transition_counts[action_index, int(source_index)]
            row.mul_(max(0.0, 1.0 - (0.2 * lr)))
            for target_strength, target_index in zip(target_weights.tolist(), target_indices, strict=False):
                hebbian_delta = lr * float(source_strength) * float(target_strength)
                row[int(target_index)] += hebbian_delta
                count_row[int(target_index)] += 1.0
            total = row.sum().clamp_min(1e-6)
            row.div_(total)
            self.reward_model[int(source_index), action_index] = (
                (1.0 - lr) * self.reward_model[int(source_index), action_index]
            ) + (lr * float(reward))

        fired_union = sorted({*source_indices, *target_indices})
        if fired_union:
            self.pool.hebbian_update_lateral(fired_union)

        self.last_prediction_error = prediction_error
        self.last_update_stats = {
            "prediction_error": float(prediction_error),
            "learning_rate_multiplier": float(reward_step.modulation.learning_rate_multiplier),
            "curiosity": float(self.compute_curiosity(next_state)),
            "intrinsic_reward": float(reward_step.reward.value),
            "novelty": float(reward_step.novelty.novelty_score),
            "exploration_drive": float(reward_step.modulation.exploration_drive),
        }
        return dict(self.last_update_stats)


__all__ = ["BioARNWorldModel", "StateRepresentation"]
