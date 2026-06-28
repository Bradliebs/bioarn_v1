"""Demo script showing Bio-ARN learning a simple RL control task."""

from __future__ import annotations

from statistics import mean

from bioarn.config import AgentConfig, RLTrainConfig, WorldModelConfig
from bioarn.training.rl_training import RLTrainer


def main() -> None:
    trainer = RLTrainer(
        RLTrainConfig(
            env_name="cartpole",
            num_episodes=40,
            max_steps_per_episode=200,
            world_model=WorldModelConfig(
                observation_dim=4,
                concept_dim=64,
                max_pool_size=40,
                num_actions=2,
                curiosity_weight=0.7,
                prediction_lr=0.08,
            ),
            agent=AgentConfig(
                epsilon_start=0.8,
                epsilon_end=0.05,
                epsilon_decay=0.98,
                curiosity_bonus=0.45,
                reward_discount=0.98,
            ),
        )
    )
    result = trainer.train()

    print("Bio-ARN RL demo (CartPole)")
    print(f"Episodes: {len(result.episode_rewards)}")
    print(f"First 10 avg reward: {mean(result.episode_rewards[:10]):.2f}")
    print(f"Last 10 avg reward: {mean(result.episode_rewards[-10:]):.2f}")
    print(f"Best 20-episode moving average: {result.best_moving_average:.2f}")
    print(f"Solved episode: {result.solved_episode}")
    print(f"Final epsilon: {result.final_epsilon:.3f}")


if __name__ == "__main__":
    main()
