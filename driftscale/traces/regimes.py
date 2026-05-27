"""Named synthetic workload regimes for Phase 1."""

from __future__ import annotations

from enum import StrEnum


class WorkloadRegime(StrEnum):
    STABLE_DIURNAL = "stable_diurnal"
    BURSTY = "bursty"
    HIGH_SUSTAINED = "high_sustained"


PHASE1_REGIMES = tuple(regime.value for regime in WorkloadRegime)

