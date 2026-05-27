"""Synthetic workload episodes used to validate the simulator before Azure traces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftscale.traces.regimes import PHASE1_REGIMES, WorkloadRegime


@dataclass(frozen=True)
class SyntheticEpisode:
    """A generated workload episode."""

    demand: np.ndarray
    regime: str
    step_minutes: int = 5
    metadata: dict[str, float | int | str] = field(default_factory=dict)


def generate_synthetic_episode(
    *,
    regime: str | WorkloadRegime = WorkloadRegime.BURSTY,
    length: int = 288,
    seed: int = 0,
    noise: float = 0.04,
    step_minutes: int = 5,
) -> SyntheticEpisode:
    """Generate a deterministic synthetic demand trace.

    The regimes mirror the MVP workload categories in the project bible: stable diurnal,
    bursty, and high sustained. Demand is expressed in abstract capacity units where one task can
    serve roughly one unit per step.
    """
    if length < 24:
        raise ValueError("length must be at least 24 steps")

    regime_value = str(regime)
    if regime_value not in PHASE1_REGIMES:
        raise ValueError(f"unknown regime {regime_value!r}; expected one of {PHASE1_REGIMES}")

    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    day_phase = 2.0 * np.pi * t / max(length, 1)

    if regime_value == WorkloadRegime.STABLE_DIURNAL:
        demand = 5.0 + 2.4 * np.sin(day_phase - 0.8) + 0.8 * np.sin(2.0 * day_phase)
    elif regime_value == WorkloadRegime.BURSTY:
        demand = 3.2 + 0.5 * np.sin(day_phase)
        centers = np.array([0.22, 0.47, 0.72]) * length
        widths = np.array([5.0, 9.0, 6.0])
        heights = np.array([8.5, 11.0, 7.5])
        for center, width, height in zip(centers, widths, heights, strict=True):
            demand += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    elif regime_value == WorkloadRegime.HIGH_SUSTAINED:
        demand = 7.5 + 1.2 * np.sin(day_phase + 0.5)
        start = int(length * 0.30)
        end = int(length * 0.78)
        demand[start:end] += 4.8
    else:  # pragma: no cover - protected by validation above
        raise AssertionError(f"unhandled regime {regime_value}")

    if noise > 0:
        demand += rng.normal(0.0, noise * max(float(np.mean(demand)), 1.0), size=length)

    demand = np.maximum(demand, 0.1).astype(np.float32)
    return SyntheticEpisode(
        demand=demand,
        regime=regime_value,
        step_minutes=step_minutes,
        metadata={
            "seed": seed,
            "length": length,
            "noise": noise,
            "mean_demand": float(np.mean(demand)),
            "p95_demand": float(np.quantile(demand, 0.95)),
        },
    )

