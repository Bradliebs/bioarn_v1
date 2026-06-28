from __future__ import annotations

import torch

from bioarn.config import AgentConfig, RLTrainConfig, WorldModelConfig
from bioarn.rl import BioARNAgent, BioARNWorldModel, CartPoleEnv, MountainCarEnv
from bioarn.training.rl_training import RLTrainer


def test_cartpole_env_step_contract() -> None:
    env = CartPoleEnv(seed=0, max_steps=25)
    observation = env.reset()

    assert observation.shape == (4,)
    next_observation, reward, done = env.step(1)

    assert next_observation.shape == (4,)
    assert reward == 1.0
    assert isinstance(done, bool)


def test_mountaincar_env_step_contract() -> None:
    env = MountainCarEnv(seed=0, max_steps=25)
    observation = env.reset()

    assert observation.shape == (2,)
    next_observation, reward, done = env.step(2)

    assert next_observation.shape == (2,)
    assert reward == -1.0
    assert isinstance(done, bool)
    assert -1.2 <= float(next_observation[0].item()) <= 0.6


def test_world_model_curiosity_tracks_novelty() -> None:
    model = BioARNWorldModel(
        WorldModelConfig(
            observation_dim=4,
            concept_dim=32,
            max_pool_size=24,
            num_actions=2,
            curiosity_weight=0.8,
        )
    )
    model.set_observation_scale(torch.tensor([2.4, 3.0, 0.21, 3.5]))

    familiar_obs = torch.tensor([0.0, 0.0, 0.0, 0.0])
    for _ in range(8):
        familiar_state = model.encode_state(familiar_obs)
    novel_state = model.encode_state(torch.tensor([1.5, 1.0, 0.18, 1.8]))

    familiar_curiosity = model.compute_curiosity(familiar_state)
    novel_curiosity = model.compute_curiosity(novel_state)
    assert novel_curiosity >= familiar_curiosity


def test_agent_selects_valid_action_and_learns() -> None:
    env = CartPoleEnv(seed=1, max_steps=50)
    agent = BioARNAgent(
        AgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            curiosity_bonus=0.4,
            reward_discount=0.95,
        ),
        world_model_config=WorldModelConfig(
            observation_dim=4,
            concept_dim=32,
            max_pool_size=32,
            num_actions=2,
        ),
        seed=1,
    )
    agent.world_model.set_observation_scale(env.observation_scale)
    observation = env.reset()

    action = agent.select_action(observation)
    next_observation, reward, done = env.step(action)
    metrics = agent.learn(observation, action, reward, next_observation, done)

    assert action in {0, 1}
    assert "prediction_error" in metrics
    assert "curiosity" in metrics
    assert agent.world_model.last_update_stats


def test_rl_trainer_improves_cartpole_rewards() -> None:
    trainer = RLTrainer(
        RLTrainConfig(
            env_name="cartpole",
            num_episodes=40,
            max_steps_per_episode=200,
            world_model=WorldModelConfig(
                observation_dim=4,
                concept_dim=48,
                max_pool_size=36,
                num_actions=2,
                curiosity_weight=0.7,
                prediction_lr=0.08,
            ),
            agent=AgentConfig(
                epsilon_start=0.6,
                epsilon_end=0.05,
                epsilon_decay=0.97,
                curiosity_bonus=0.45,
                reward_discount=0.98,
            ),
        )
    )

    result = trainer.train(num_episodes=40)

    early = sum(result.episode_rewards[:10]) / 10.0
    late = sum(result.episode_rewards[-10:]) / 10.0
    assert result.best_moving_average >= early
    assert late >= early
    assert result.final_epsilon <= 0.6
    assert len(result.mean_curiosity) == 40
