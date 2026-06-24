"""Hebbian-style vote reweighting for ensemble experts."""

from __future__ import annotations

import torch


class HebbianBoosting:
    """Boost expert-class associations when they help identify the right class."""

    def __init__(
        self,
        num_experts: int,
        num_classes: int,
        *,
        learning_rate: float = 0.2,
        penalty_rate: float = 0.08,
        min_weight: float = 0.2,
        max_weight: float = 4.0,
    ) -> None:
        self.num_experts = int(max(1, num_experts))
        self.num_classes = int(max(1, num_classes))
        self.learning_rate = float(learning_rate)
        self.penalty_rate = float(penalty_rate)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.weights = torch.ones(self.num_experts, self.num_classes, dtype=torch.float32)

    def ensure_capacity(self, num_experts: int, num_classes: int) -> None:
        target_experts = int(max(1, num_experts))
        target_classes = int(max(1, num_classes))
        if target_experts <= self.num_experts and target_classes <= self.num_classes:
            return

        expanded = torch.ones(target_experts, target_classes, dtype=torch.float32)
        expanded[: self.num_experts, : self.num_classes] = self.weights
        self.weights = expanded
        self.num_experts = target_experts
        self.num_classes = target_classes

    def update_weights(self, expert_predictions: list[int | None], true_label: int) -> None:
        class_index = int(true_label)
        self.ensure_capacity(len(expert_predictions), class_index + 1)

        for expert_index, prediction in enumerate(expert_predictions):
            if prediction is None or int(prediction) < 0:
                self.weights[expert_index, class_index].mul_(1.0 - (self.penalty_rate * 0.5))
                continue

            predicted = int(prediction)
            if predicted == class_index:
                self.weights[expert_index, class_index].add_(self.learning_rate)
            else:
                self.weights[expert_index, class_index].mul_(1.0 - self.penalty_rate)
                if predicted < self.num_classes:
                    self.weights[expert_index, predicted].mul_(1.0 - (self.penalty_rate * 0.5))

        self.weights.clamp_(min=self.min_weight, max=self.max_weight)
        column_means = self.weights.mean(dim=0, keepdim=True).clamp_min(1e-6)
        self.weights.div_(column_means)
        self.weights.clamp_(min=self.min_weight, max=self.max_weight)

    def get_weights(self, expert_idx: int) -> torch.Tensor:
        index = int(expert_idx)
        if index < 0 or index >= self.num_experts:
            raise IndexError("expert_idx out of range.")
        return self.weights[index].detach().clone()


__all__ = ["HebbianBoosting"]
