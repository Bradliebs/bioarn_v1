"""Standalone classic-control environments without a gym dependency."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch


class SimpleEnvironment(ABC):
    """Lightweight environment interface used by the Bio-ARN RL trainer."""

    observation_dim: int
    num_actions: int

    def __init__(self, *, seed: int = 0, max_steps: int = 500) -> None:
        self.seed = int(seed)
        self.max_steps = int(max(1, max_steps))
        self.generator = torch.Generator().manual_seed(self.seed)
        self.steps = 0

    @property
    @abstractmethod
    def observation_scale(self) -> torch.Tensor:
        """Characteristic observation scales used for world-model normalization."""

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """Advance the environment by one step."""

    def sample_action(self) -> int:
        return int(torch.randint(self.num_actions, (1,), generator=self.generator).item())

    def _uniform(self, low: float, high: float, size: int) -> torch.Tensor:
        return low + ((high - low) * torch.rand(size, generator=self.generator, dtype=torch.float32))


class CartPoleEnv(SimpleEnvironment):
    """Classic CartPole swing-balancing task implemented from scratch."""

    observation_dim = 4
    num_actions = 2

    def __init__(self, *, seed: int = 0, max_steps: int = 500) -> None:
        super().__init__(seed=seed, max_steps=max_steps)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
        self.x_threshold = 2.4
        self.state = torch.zeros(self.observation_dim, dtype=torch.float32)

    @property
    def observation_scale(self) -> torch.Tensor:
        return torch.tensor(
            [self.x_threshold, 3.0, self.theta_threshold_radians, 3.5],
            dtype=torch.float32,
        )

    def reset(self) -> torch.Tensor:
        self.steps = 0
        self.state = self._uniform(-0.05, 0.05, self.observation_dim)
        return self.state.clone()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        if int(action) not in {0, 1}:
            raise ValueError(f"CartPole expects action 0 or 1, got {action}.")

        x, x_dot, theta, theta_dot = [float(value) for value in self.state.tolist()]
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * (theta_dot**2) * sintheta) / self.total_mass
        theta_acc = (
            self.gravity * sintheta - costheta * temp
        ) / (self.length * ((4.0 / 3.0) - ((self.masspole * (costheta**2)) / self.total_mass)))
        x_acc = temp - ((self.polemass_length * theta_acc * costheta) / self.total_mass)

        x = x + (self.tau * x_dot)
        x_dot = x_dot + (self.tau * x_acc)
        theta = theta + (self.tau * theta_dot)
        theta_dot = theta_dot + (self.tau * theta_acc)
        self.state = torch.tensor([x, x_dot, theta, theta_dot], dtype=torch.float32)

        self.steps += 1
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )
        reward = 1.0
        return self.state.clone(), reward, terminated


class MountainCarEnv(SimpleEnvironment):
    """Classic MountainCar task implemented from scratch."""

    observation_dim = 2
    num_actions = 3

    def __init__(self, *, seed: int = 0, max_steps: int = 200) -> None:
        super().__init__(seed=seed, max_steps=max_steps)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0
        self.force = 0.001
        self.gravity = 0.0025
        self.state = torch.zeros(self.observation_dim, dtype=torch.float32)

    @property
    def observation_scale(self) -> torch.Tensor:
        return torch.tensor([1.0, self.max_speed], dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        self.steps = 0
        position = float(self._uniform(-0.6, -0.4, 1).item())
        self.state = torch.tensor([position, 0.0], dtype=torch.float32)
        return self.state.clone()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        if int(action) not in {0, 1, 2}:
            raise ValueError(f"MountainCar expects actions 0, 1, 2, got {action}.")

        position, velocity = [float(value) for value in self.state.tolist()]
        force = float(int(action) - 1)
        velocity += (force * self.force) - (self.gravity * math.cos(3.0 * position))
        velocity = min(max(velocity, -self.max_speed), self.max_speed)
        position += velocity
        position = min(max(position, self.min_position), self.max_position)
        if position <= self.min_position and velocity < 0.0:
            velocity = 0.0

        self.state = torch.tensor([position, velocity], dtype=torch.float32)
        self.steps += 1
        terminated = bool(
            (position >= self.goal_position and velocity >= self.goal_velocity)
            or self.steps >= self.max_steps
        )
        reward = -1.0
        return self.state.clone(), reward, terminated


__all__ = ["CartPoleEnv", "MountainCarEnv", "SimpleEnvironment"]
