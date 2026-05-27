"""Vectorized trace preprocessing and trace-to-demand mappings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd


class DemandMappingVariant(StrEnum):
    """Supported §5.5 trace-to-demand mapping variants."""

    LINEAR = "linear"
    CONVEX = "convex"
    THRESHOLD = "threshold"


@dataclass(frozen=True)
class DemandMappingConfig:
    """Configuration for mapping aligned VM CPU readings into demand units."""

    variant: str = DemandMappingVariant.LINEAR
    scale_factor: float = 1.0
    alpha: float = 2.0
    spike_threshold: float = 0.90


def compute_demand_signal(
    cpu_matrix: pd.DataFrame | np.ndarray,
    *,
    variant: str | DemandMappingVariant = DemandMappingVariant.LINEAR,
    scale_factor: float = 1.0,
    alpha: float = 2.0,
    spike_threshold: float = 0.90,
) -> np.ndarray:
    """Map an aligned CPU matrix into a demand vector using vectorized numpy operations."""
    values = _as_cpu_array(cpu_matrix)
    variant_value = DemandMappingVariant(str(variant))

    if variant_value == DemandMappingVariant.LINEAR:
        demand = scale_factor * np.nansum(values, axis=1)
    elif variant_value == DemandMappingVariant.CONVEX:
        demand = scale_factor * np.nansum(np.power(values, 1.5), axis=1)
    elif variant_value == DemandMappingVariant.THRESHOLD:
        spike_bonus = alpha * np.any(values > spike_threshold, axis=1).astype(np.float32)
        demand = (scale_factor * np.nansum(values, axis=1)) + spike_bonus
    else:  # pragma: no cover - StrEnum construction validates this path.
        raise AssertionError(f"unhandled mapping variant {variant_value}")

    return demand.astype(np.float32)


def build_preprocessed_trace(
    cpu_matrix: pd.DataFrame,
    *,
    mapping_config: DemandMappingConfig | None = None,
) -> pd.DataFrame:
    """Create the lightweight cached trace consumed by later DriftScaleEnv runs."""
    config = mapping_config or DemandMappingConfig()
    values = _as_cpu_array(cpu_matrix)
    demand = compute_demand_signal(
        values,
        variant=config.variant,
        scale_factor=config.scale_factor,
        alpha=config.alpha,
        spike_threshold=config.spike_threshold,
    )

    aggregate_cpu = np.nansum(values, axis=1).astype(np.float32)
    max_cpu = np.nanmax(values, axis=1).astype(np.float32)
    mean_cpu = np.nanmean(values, axis=1).astype(np.float32)
    active_vm_count = np.sum(~np.isnan(values), axis=1).astype(np.int32)
    spike_any = (max_cpu > config.spike_threshold).astype(np.int8)

    timestamps = cpu_matrix.index
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "demand": demand,
            "aggregate_cpu": aggregate_cpu,
            "max_cpu": max_cpu,
            "mean_cpu": mean_cpu,
            "active_vm_count": active_vm_count,
            "spike_any": spike_any,
            "mapping_variant": str(config.variant),
            "scale_factor": float(config.scale_factor),
            "alpha": float(config.alpha),
        }
    )


def write_preprocessed_trace(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a cached preprocessed trace as CSV or parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("output_path must end in .csv or .parquet")

    return path


def load_preprocessed_demand(path: str | Path, *, demand_column: str = "demand") -> np.ndarray:
    """Load a cached demand signal for constructing DriftScaleEnv."""
    cache_path = Path(path)
    if cache_path.suffix.lower() == ".csv":
        frame = pd.read_csv(cache_path)
    elif cache_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(cache_path)
    else:
        raise ValueError("preprocessed trace must be .csv or .parquet")

    if demand_column not in frame.columns:
        raise ValueError(f"{demand_column!r} column missing from {cache_path}")
    return pd.to_numeric(frame[demand_column], errors="raise").to_numpy(dtype=np.float32)


def _as_cpu_array(cpu_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(cpu_matrix, pd.DataFrame):
        values = cpu_matrix.to_numpy(dtype=np.float32)
    else:
        values = cpu_matrix
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("cpu_matrix must be two-dimensional: timesteps x VMs")
    return np.clip(array, 0.0, 1.0)
