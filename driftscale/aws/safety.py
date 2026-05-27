"""Safety guardrails for the live ECS controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass(frozen=True)
class SafetyConfig:
    min_tasks: int = 1
    max_tasks: int = 6
    cooldown_seconds: int = 60
    max_scale_delta: int = 1


@dataclass(frozen=True)
class SafetyDecision:
    current_count: int
    proposed_delta: int
    bounded_delta: int
    desired_count: int
    should_update: bool
    intervention: str


class EcsSafetyWrapper:
    """Clamp policy actions before any live ECS update_service call."""

    def __init__(self, config: SafetyConfig) -> None:
        if config.min_tasks < 1 or config.max_tasks < config.min_tasks:
            raise ValueError("safety bounds must satisfy 1 <= min_tasks <= max_tasks")
        if config.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if config.max_scale_delta < 0:
            raise ValueError("max_scale_delta must be non-negative")
        self.config = config
        self.last_scale_time: datetime | None = None

    def apply(
        self,
        *,
        current_count: int,
        proposed_delta: int,
        now: datetime | None = None,
    ) -> SafetyDecision:
        now = now or datetime.now(UTC)
        current_count = int(np.clip(current_count, self.config.min_tasks, self.config.max_tasks))
        proposed_delta = int(proposed_delta)
        bounded_delta = int(
            np.clip(proposed_delta, -self.config.max_scale_delta, self.config.max_scale_delta)
        )
        desired_count = int(
            np.clip(current_count + bounded_delta, self.config.min_tasks, self.config.max_tasks)
        )
        interventions = []

        if bounded_delta != proposed_delta:
            interventions.append("max_delta")
        if desired_count != current_count + bounded_delta:
            interventions.append("bounds")

        if desired_count != current_count and self._in_cooldown(now):
            desired_count = current_count
            bounded_delta = 0
            interventions.append("cooldown")

        should_update = desired_count != current_count
        if should_update:
            self.last_scale_time = now

        return SafetyDecision(
            current_count=current_count,
            proposed_delta=proposed_delta,
            bounded_delta=bounded_delta,
            desired_count=desired_count,
            should_update=should_update,
            intervention="none" if not interventions else ",".join(interventions),
        )

    def _in_cooldown(self, now: datetime) -> bool:
        if self.last_scale_time is None:
            return False
        elapsed = (now - self.last_scale_time).total_seconds()
        return elapsed < self.config.cooldown_seconds
