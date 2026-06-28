"""Training loop for Bio-ARN reinforcement-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from bioarn.config import RLTrainConfig, WorldModelConfig
from bioarn.rl import BioARNAgent, CartPoleEnv, MountainCarEnv


@dataclass
class TrainingResult:
    """Episode-level summary for an RL training run."""

    env_name: str
    episode_rewards: list[float]
    episode_lengths: list[int]
    moving_average_rewards: list[float]
    mean_curiosity: list[float]
    mean_prediction_error: list[float]
    solved_episode: int | None
    final_epsilon: float
    best_moving_average: float


class RLTrainer:
    """Train a Bio-ARN agent on standalone RL environments."""

    def __init__(self, config: RLTrainConfig):
        self.config = config
        torch.manual_seed(0)
        self.env = self._build_environment(config.env_name, config.max_steps_per_episode)
        world_model_config = replace(
            config.world_model or WorldModelConfig(),
            observation_dim=self.env.observation_dim,
            num_actions=self.env.num_actions,
            max_pool_size=max(8, int((config.world_model or WorldModelConfig()).max_pool_size)),
        )
        self.agent = BioARNAgent(
            config.agent,
            world_model_config=world_model_config,
            seed=0,
        )
        self.agent.world_model.set_observation_scale(self.env.observation_scale)

    @staticmethod
    def _build_environment(env_name: str, max_steps: int):
        name = env_name.strip().lower()
        if name == "cartpole":
            return CartPoleEnv(seed=0, max_steps=max_steps)
        if name == "mountaincar":
            return MountainCarEnv(seed=0, max_steps=max_steps)
        raise ValueError(f"Unsupported RL environment: {env_name}")

    def train(self, num_episodes: int | None = None) -> TrainingResult:
        """Train the agent and return episode-level metrics."""

        total_episodes = int(max(1, self.config.num_episodes if num_episodes is None else num_episodes))
        rewards: list[float] = []
        lengths: list[int] = []
        moving_average: list[float] = []
        curiosity: list[float] = []
        prediction_errors: list[float] = []
        solved_episode: int | None = None
        solve_threshold = 195.0 if self.config.env_name == "cartpole" else -110.0

        for episode in range(total_episodes):
            result = self.agent.train_episode(self.env)
            rewards.append(float(result.total_reward))
            lengths.append(int(result.steps))
            curiosity.append(float(result.mean_curiosity))
            prediction_errors.append(float(result.mean_prediction_error))
            recent = rewards[-20:]
            average_reward = sum(recent) / len(recent)
            moving_average.append(average_reward)
            if solved_episode is None and average_reward >= solve_threshold:
                solved_episode = episode + 1

        return TrainingResult(
            env_name=self.config.env_name,
            episode_rewards=rewards,
            episode_lengths=lengths,
            moving_average_rewards=moving_average,
            mean_curiosity=curiosity,
            mean_prediction_error=prediction_errors,
            solved_episode=solved_episode,
            final_epsilon=float(self.agent.epsilon),
            best_moving_average=max(moving_average, default=0.0),
        )


__all__ = ["RLTrainer", "TrainingResult"]
