"""Gymnasium environment for trace-driven autoscaling simulation."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driftscale.envs.reward import RewardConfig, calculate_reward
from driftscale.envs.spaces import (
    ACTION_DELTAS,
    action_mask,
    build_observation_space,
    normalize_action_delta,
    normalize_task_count,
)


class DriftScaleEnv(gym.Env):
    """Autoscaling simulator over a fixed demand episode.

    The environment exposes the twelve normalized features from the project bible and a
    five-action discrete scale delta. Boundary actions are surfaced through ``action_masks`` so
    MaskablePPO can avoid invalid actions in later phases.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand: np.ndarray | list[float],
        *,
        min_tasks: int = 1,
        max_tasks: int = 20,
        initial_tasks: int | None = None,
        capacity_per_task: float = 1.0,
        reward_config: RewardConfig | None = None,
        step_minutes: int = 5,
        strict_action_mask: bool = True,
    ) -> None:
        super().__init__()
        demand_array = np.asarray(demand, dtype=np.float32)
        if demand_array.ndim != 1 or demand_array.size < 2:
            raise ValueError("demand must be a one-dimensional sequence with at least two steps")
        if np.any(demand_array < 0):
            raise ValueError("demand must be non-negative")
        if min_tasks < 1 or max_tasks < min_tasks:
            raise ValueError("task bounds must satisfy 1 <= min_tasks <= max_tasks")
        if capacity_per_task <= 0:
            raise ValueError("capacity_per_task must be positive")

        self.demand = demand_array
        self.min_tasks = int(min_tasks)
        self.max_tasks = int(max_tasks)
        self.initial_tasks = int(initial_tasks or min_tasks)
        if not self.min_tasks <= self.initial_tasks <= self.max_tasks:
            raise ValueError("initial_tasks must be inside task bounds")

        self.capacity_per_task = float(capacity_per_task)
        self.reward_config = reward_config or RewardConfig()
        self.step_minutes = int(step_minutes)
        self.strict_action_mask = strict_action_mask

        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.observation_space = build_observation_space()

        self._index = 0
        self._task_count = self.initial_tasks
        self._previous_delta = 0
        self._cpu_history: deque[float] = deque(maxlen=60)
        self._slo_history: deque[float] = deque(maxlen=15)
        self._scale_history: deque[float] = deque(maxlen=15)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._index = 0
        self._task_count = int((options or {}).get("initial_tasks", self.initial_tasks))
        self._previous_delta = 0

        current_util = self._utilization_for(self._index, self._task_count)
        self._cpu_history = deque([current_util] * 60, maxlen=60)
        self._slo_history = deque([0.0] * 15, maxlen=15)
        self._scale_history = deque([0.0] * 15, maxlen=15)

        return self._observation(), self._info_for_current_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= len(ACTION_DELTAS):
            raise ValueError(f"action must be in [0, {len(ACTION_DELTAS) - 1}]")

        if self.strict_action_mask and not self.action_masks()[action]:
            raise ValueError(
                f"action {action} with delta {ACTION_DELTAS[action]} is invalid "
                f"for task_count={self._task_count}"
            )

        old_task_count = self._task_count
        delta = int(ACTION_DELTAS[action])
        self._task_count = int(np.clip(old_task_count + delta, self.min_tasks, self.max_tasks))
        applied_delta = self._task_count - old_task_count
        capacity = self._task_count * self.capacity_per_task
        demand = float(self.demand[self._index])
        breakdown = calculate_reward(
            demand=demand,
            capacity=capacity,
            old_task_count=old_task_count,
            new_task_count=self._task_count,
            config=self.reward_config,
        )

        utilization = min(demand / max(capacity, self.reward_config.eps), 1.0)
        self._cpu_history.append(utilization)
        self._slo_history.append(float(breakdown.slo_violation))
        self._scale_history.append(float(applied_delta != 0))
        self._previous_delta = applied_delta

        info = {
            **self._info_for_current_state(),
            "old_task_count": old_task_count,
            "action_delta": applied_delta,
            "resource_cost": breakdown.resource_cost,
            "slo_penalty": breakdown.slo_penalty,
            "scaling_action_penalty": breakdown.scaling_action_penalty,
            "overprovision_penalty": breakdown.overprovision_penalty,
            "total_penalty": breakdown.total_penalty,
            "overprovision_ratio": breakdown.overprovision_ratio,
        }

        self._index += 1
        terminated = self._index >= len(self.demand)
        truncated = False
        return self._observation(), breakdown.reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask compatible with sb3-contrib MaskablePPO."""
        return action_mask(self._task_count, self.min_tasks, self.max_tasks)

    @property
    def task_count(self) -> int:
        return self._task_count

    @property
    def index(self) -> int:
        return self._index

    def _observation(self) -> np.ndarray:
        current_index = min(self._index, len(self.demand) - 1)
        current_util = self._utilization_for(current_index, self._task_count)
        cpu_values = list(self._cpu_history)

        step_of_day = ((current_index * self.step_minutes) % (24 * 60)) / (24 * 60)
        angle = 2.0 * np.pi * step_of_day

        observation = np.array(
            [
                current_util,
                float(np.mean(cpu_values[-5:])),
                float(np.mean(cpu_values[-15:])),
                float(np.mean(cpu_values[-60:])),
                float(np.max(cpu_values[-15:])),
                float(np.std(cpu_values[-15:])),
                normalize_task_count(self._task_count, self.min_tasks, self.max_tasks),
                normalize_action_delta(self._previous_delta),
                float(np.sin(angle)),
                float(np.cos(angle)),
                float(np.mean(self._slo_history)),
                float(np.mean(self._scale_history)),
            ],
            dtype=np.float32,
        )
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _utilization_for(self, index: int, task_count: int) -> float:
        capacity = task_count * self.capacity_per_task
        return float(min(self.demand[index] / max(capacity, self.reward_config.eps), 1.0))

    def _info_for_current_state(self) -> dict[str, Any]:
        current_index = min(self._index, len(self.demand) - 1)
        demand = float(self.demand[current_index])
        capacity = self._task_count * self.capacity_per_task
        return {
            "step": self._index,
            "demand": demand,
            "capacity": capacity,
            "task_count": self._task_count,
            "cpu_utilization": min(demand / max(capacity, self.reward_config.eps), 1.0),
            "slo_violation": demand > capacity,
            "valid_action_mask": self.action_masks(),
        }

