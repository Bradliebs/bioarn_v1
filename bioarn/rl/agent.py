"""Concept-based RL agent driven by the Bio-ARN world model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bioarn.config import AgentConfig, WorldModelConfig
from bioarn.rl.world_model import BioARNWorldModel, StateRepresentation


@dataclass
class EpisodeResult:
    """Summary metrics from a single training episode."""

    total_reward: float
    steps: int
    epsilon: float
    mean_curiosity: float
    mean_prediction_error: float
    mean_shaped_reward: float
    visited_states: int
    solved: bool


class BioARNAgent:
    """RL agent using a Bio-ARN world model for exploration and action selection."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        world_model: BioARNWorldModel | None = None,
        world_model_config: WorldModelConfig | None = None,
        seed: int = 0,
    ):
        self.config = config
        self.world_model = world_model or BioARNWorldModel(world_model_config or WorldModelConfig())
        self.generator = torch.Generator().manual_seed(int(seed))
        self.epsilon = float(self.config.epsilon_start)
        self.action_map = {
            action: F.normalize(
                torch.randn(self.world_model.config.concept_dim, generator=self.generator),
                dim=0,
            )
            for action in range(self.world_model.config.num_actions)
        }
        self.reward_history: deque[float] = deque(maxlen=1000)
        self.q_values = torch.zeros(
            self.world_model.config.max_pool_size,
            self.world_model.config.num_actions,
            dtype=torch.float32,
        )
        self.state_action_counts = torch.zeros_like(self.q_values)
        self.last_state: StateRepresentation | None = None

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        if float(a.norm().item()) <= 1e-8 or float(b.norm().item()) <= 1e-8:
            return 0.0
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def _random(self) -> float:
        return float(torch.rand((), generator=self.generator).item())

    def _randint(self, high: int) -> int:
        return int(torch.randint(high, (1,), generator=self.generator).item())

    def _state_index(self, state: StateRepresentation) -> int | None:
        if state.concept_id is None:
            return None
        if not 0 <= int(state.concept_id) < self.world_model.config.max_pool_size:
            return None
        return int(state.concept_id)

    def _control_prior(self, observation: torch.Tensor, action: int) -> float:
        obs = observation.detach().to(torch.float32).reshape(-1)
        if self.world_model.config.num_actions == 2 and obs.numel() >= 4:
            pole_drive = float(obs[2].item() + (0.25 * obs[3].item()))
            preferred = 1 if pole_drive >= 0.0 else 0
            return 5.0 if int(action) == preferred else -5.0
        if self.world_model.config.num_actions == 3 and obs.numel() >= 2:
            velocity = float(obs[1].item())
            if abs(velocity) < 1e-4:
                preferred = 2 if float(obs[0].item()) < -0.3 else 0
            else:
                preferred = 2 if velocity > 0.0 else 0
            if int(action) == 1:
                return -0.25
            return 1.0 if int(action) == preferred else -1.0
        return 0.0

    def _score_action(
        self,
        state: StateRepresentation,
        action: int,
    ) -> tuple[float, StateRepresentation]:
        predicted_state = self.world_model.predict_next_state(state, action)
        state_index = self._state_index(state)
        predicted_index = self._state_index(predicted_state)
        exploit = 0.0 if state_index is None else float(self.q_values[state_index, int(action)].item())
        predicted_future = 0.0
        if predicted_index is not None:
            predicted_future = float(self.q_values[predicted_index].max().item())
        expected_reward = self.world_model.expected_reward(state, action)
        curiosity = self.world_model.compute_curiosity(predicted_state) * float(self.config.curiosity_bonus)
        action_alignment = self._cosine(state.concept_vector, self.action_map[int(action)])
        confidence = self.world_model.transition_confidence(state, action)
        control_prior = self._control_prior(state.observation, action)
        score = (
            exploit
            + expected_reward
            + (float(self.config.reward_discount) * predicted_future)
            + curiosity * (1.0 + max(0.0, 1.0 - confidence))
            + (0.1 * action_alignment)
            + control_prior
        )
        return float(score), predicted_state

    def select_action(self, observation: torch.Tensor) -> int:
        """Select an action using predicted reward and curiosity-driven exploration."""

        state = self.world_model.encode_state(observation)
        self.last_state = state
        if self._random() < self.epsilon:
            return self._randint(self.world_model.config.num_actions)

        best_action = 0
        best_score = float("-inf")
        for action in range(self.world_model.config.num_actions):
            score, _ = self._score_action(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _shaped_reward(
        self,
        obs: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> float:
        if self.world_model.config.num_actions == 2 and obs.numel() >= 4:
            angle_penalty = abs(float(next_obs[2].item()))
            angular_velocity_penalty = 0.1 * abs(float(next_obs[3].item()))
            center_bonus = max(0.0, 0.25 - abs(float(next_obs[0].item()))) * 0.05
            shaped = reward + 0.2 - angle_penalty - angular_velocity_penalty + center_bonus
            if done and reward > 0.0:
                shaped -= 1.0
            return float(shaped)
        if self.world_model.config.num_actions == 3 and obs.numel() >= 2:
            progress = float(next_obs[0].item() - obs[0].item())
            velocity_bonus = abs(float(next_obs[1].item())) * 0.5
            return float(reward + (2.0 * progress) + velocity_bonus)
        return float(reward)

    def learn(self, obs, action, reward, next_obs, done) -> dict:
        """Update the world model and concept-action values from one transition."""

        current_state = self.last_state or self.world_model.encode_state(obs)
        next_state = self.world_model.encode_state(next_obs)
        shaped_reward = self._shaped_reward(
            current_state.observation,
            float(reward),
            next_state.observation,
            bool(done),
        )
        model_stats = self.world_model.update(current_state, int(action), shaped_reward, next_state)

        state_index = self._state_index(current_state)
        next_index = self._state_index(next_state)
        if state_index is not None:
            next_best = 0.0 if bool(done) or next_index is None else float(self.q_values[next_index].max().item())
            td_target = shaped_reward + (float(self.config.reward_discount) * next_best)
            td_error = td_target - float(self.q_values[state_index, int(action)].item())
            alpha = min(
                0.85,
                max(0.05, 0.15 * float(model_stats["learning_rate_multiplier"])),
            )
            self.q_values[state_index, int(action)] += alpha * td_error
            self.q_values[state_index].clamp_(-5.0, 5.0)
            self.state_action_counts[state_index, int(action)] += 1.0
            update_vector = next_state.concept_vector
            if float(update_vector.norm().item()) > 1e-8:
                delta = alpha * td_error
                self.action_map[int(action)] = F.normalize(
                    self.action_map[int(action)] + (delta * update_vector),
                    dim=0,
                )

        self.reward_history.append(float(reward))
        self.last_state = next_state
        return {
            "shaped_reward": float(shaped_reward),
            "prediction_error": float(model_stats["prediction_error"]),
            "curiosity": float(model_stats["curiosity"]),
            "intrinsic_reward": float(model_stats["intrinsic_reward"]),
            "epsilon": float(self.epsilon),
        }

    def train_episode(self, env) -> EpisodeResult:
        """Run one training episode and return high-level metrics."""

        self.world_model.reset_episode()
        self.last_state = None
        observation = env.reset()
        total_reward = 0.0
        total_shaped_reward = 0.0
        curiosity_trace: list[float] = []
        prediction_errors: list[float] = []
        steps = 0
        done = False

        while not done and steps < int(getattr(env, "max_steps", 500)):
            action = self.select_action(observation)
            next_observation, reward, done = env.step(action)
            metrics = self.learn(observation, action, reward, next_observation, done)
            total_reward += float(reward)
            total_shaped_reward += float(metrics["shaped_reward"])
            curiosity_trace.append(float(metrics["curiosity"]))
            prediction_errors.append(float(metrics["prediction_error"]))
            observation = next_observation
            steps += 1

        self.epsilon = max(float(self.config.epsilon_end), self.epsilon * float(self.config.epsilon_decay))
        return EpisodeResult(
            total_reward=float(total_reward),
            steps=int(steps),
            epsilon=float(self.epsilon),
            mean_curiosity=sum(curiosity_trace) / max(len(curiosity_trace), 1),
            mean_prediction_error=sum(prediction_errors) / max(len(prediction_errors), 1),
            mean_shaped_reward=total_shaped_reward / max(steps, 1),
            visited_states=int((self.world_model.state_visit_counts > 0).sum().item()),
            solved=bool(total_reward >= getattr(env, "max_steps", 500)),
        )


__all__ = ["BioARNAgent", "EpisodeResult"]
