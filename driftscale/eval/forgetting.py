"""Forgetting and backward-transfer metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForgettingMetrics:
    """Task-A retention after training on Task B."""

    method: str
    task_a_reward_before_b: float
    task_a_reward_after_b: float

    @property
    def backward_transfer(self) -> float:
        """BWT: old-task reward after new-task training minus old-task reward before it."""
        return self.task_a_reward_after_b - self.task_a_reward_before_b

    @property
    def forgetting(self) -> float:
        """Positive values mean Task-A reward decreased after Task-B training."""
        return max(0.0, -self.backward_transfer)

    @property
    def retention_ratio(self) -> float:
        denominator = abs(self.task_a_reward_before_b)
        if denominator == 0.0:
            return 1.0
        return max(0.0, 1.0 - (self.forgetting / denominator))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "task_a_reward_before_b": self.task_a_reward_before_b,
            "task_a_reward_after_b": self.task_a_reward_after_b,
            "bwt": self.backward_transfer,
            "forgetting": self.forgetting,
            "retention_ratio": self.retention_ratio,
        }


def calculate_bwt(*, reward_before: float, reward_after: float) -> float:
    """Return backward transfer from two old-task reward measurements."""
    return reward_after - reward_before
