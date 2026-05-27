"""Azure VM trace loading utilities.

The public Azure traces have appeared in slightly different CSV shapes across mirrors and
preprocessed samples. This loader accepts explicit column names when known, but also detects common
``timestamp``, ``vm_id``, and CPU-utilization aliases for small local extracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_AZURE_V1_CPU_PATH = Path("data/raw/vm_cpu_readings-file-1-of-125.csv.gz")
DEFAULT_AZURE_RAW_DIR = Path("data/raw")
AZURE_CHECKPOINT_IDS = (1, 25, 50, 75, 100, 125)
AZURE_V1_CPU_COLUMNS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]


@dataclass(frozen=True)
class AzureTraceColumns:
    """Column names used by a raw Azure VM CPU CSV."""

    timestamp: str = "timestamp"
    vm_id: str = "vm_id"
    cpu: str = "avg_cpu"


@dataclass(frozen=True)
class ChronologicalSplit:
    """Dense Azure CPU matrix split into chronological Task A and Task B windows."""

    task_a: pd.DataFrame
    task_b: pd.DataFrame
    selected_vms: list[str]


class VmSelectionStrategy(StrEnum):
    """Checkpoint VM cohort selection modes for Azure curricula."""

    PERSISTENT_DENSE = "persistent_dense"
    PER_CHECKPOINT_DENSE = "per_checkpoint_dense"


@dataclass(frozen=True)
class AzureCheckpointRegimes:
    """Sequential dense CPU matrices for Azure checkpoint shards."""

    checkpoint_ids: list[int]
    paths: list[Path]
    selected_vms: list[str]
    selected_vms_by_checkpoint: list[list[str]]
    matrices: list[pd.DataFrame]
    selection_strategy: str = VmSelectionStrategy.PER_CHECKPOINT_DENSE


_COLUMN_ALIASES = {
    "timestamp": (
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "sample_time",
        "measurement_time",
        "minute",
        "interval",
    ),
    "vm_id": (
        "vm_id",
        "vmid",
        "vm",
        "vm id",
        "machine_id",
        "machineid",
        "instance_id",
        "deployment_id",
    ),
    "cpu": (
        "avg_cpu",
        "average_cpu",
        "cpu",
        "cpu_utilization",
        "cpu_utilisation",
        "cpu_percent",
        "cpu_percentage",
        "cpu_usage",
        "avgcpu",
        "max_cpu",
    ),
}


def load_azure_cpu_matrix(
    raw_csv_path: str | Path,
    *,
    vm_count: int = 500,
    columns: AzureTraceColumns | None = None,
    has_header: bool = True,
    column_names: list[str] | None = None,
    max_rows: int | None = None,
    chunksize: int | None = None,
    fill_method: str = "interpolate",
    timestamp_unit: str | None = None,
) -> pd.DataFrame:
    """Load a raw Azure VM CPU CSV into a timestamp-aligned CPU matrix.

    Returns a dataframe indexed by timestamp and with one selected VM per column. CPU values are
    normalized to ``[0, 1]`` whether the raw file uses fractions or percentages.
    """
    if vm_count <= 0:
        raise ValueError("vm_count must be positive")

    if chunksize:
        chunks = []
        for raw_chunk in _read_raw_csv_chunks(
            raw_csv_path,
            has_header=has_header,
            column_names=column_names,
            max_rows=max_rows,
            chunksize=chunksize,
        ):
            resolved_columns = columns or detect_azure_columns(raw_chunk.columns)
            chunks.append(
                _compact_cpu_frame(
                    raw_chunk,
                    columns=resolved_columns,
                    timestamp_unit=timestamp_unit,
                )
            )
        if not chunks:
            raise ValueError("no rows were read from trace")
        compact = pd.concat(chunks, ignore_index=True)
    else:
        raw = _read_raw_csv(
            raw_csv_path,
            has_header=has_header,
            column_names=column_names,
            max_rows=max_rows,
        )
        resolved_columns = columns or detect_azure_columns(raw.columns)
        compact = _compact_cpu_frame(raw, columns=resolved_columns, timestamp_unit=timestamp_unit)

    selected_vms = select_vm_subset(compact, vm_count=vm_count)
    filtered = compact[compact["vm_id"].isin(selected_vms)]
    if filtered.empty:
        raise ValueError("no rows remain after VM subset filtering")

    matrix = filtered.pivot_table(
        index="timestamp",
        columns="vm_id",
        values="cpu",
        aggfunc="mean",
    ).sort_index()
    matrix = matrix.reindex(columns=selected_vms)
    return align_cpu_matrix(matrix, fill_method=fill_method)


def load_dense_azure_cpu_matrix(
    raw_csv_path: str | Path = DEFAULT_AZURE_V1_CPU_PATH,
    *,
    vm_count: int = 1000,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the real headerless Azure V1 CPU shard as a dense top-VM matrix.

    The file is read in two passes to avoid OOM:
    1. Count VM observations and select the densest ``vm_count`` VMs.
    2. Filter to those VMs and pivot into timestamp x vm_id.

    Missing 5-minute ticks are handled exactly as requested:
    ``.fillna(method='ffill').fillna(0)``.
    """
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if vm_count <= 0:
        raise ValueError("vm_count must be positive")

    selected_vms = select_dense_vms(
        path,
        vm_count=vm_count,
        chunksize=chunksize,
        max_rows=max_rows,
    )
    return load_persistent_vm_matrix(
        path,
        selected_vms=selected_vms,
        chunksize=chunksize,
        max_rows=max_rows,
    )


def select_dense_vms(
    raw_csv_path: str | Path = DEFAULT_AZURE_V1_CPU_PATH,
    *,
    vm_count: int = 1000,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> list[str]:
    """Return the VM IDs with the most readings in the real Azure V1 CPU shard."""
    path = Path(raw_csv_path)
    counts: pd.Series | None = None
    rows_seen = 0
    for chunk in pd.read_csv(
        path,
        compression="infer",
        header=None,
        names=AZURE_V1_CPU_COLUMNS,
        usecols=["vm_id"],
        chunksize=chunksize,
    ):
        rows_seen += len(chunk)
        chunk_counts = chunk["vm_id"].astype(str).value_counts()
        counts = chunk_counts if counts is None else counts.add(chunk_counts, fill_value=0)
        if max_rows is not None and rows_seen >= max_rows:
            break

    if counts is None or counts.empty:
        raise ValueError("no VM IDs found in Azure CPU shard")
    return counts.sort_values(ascending=False).head(vm_count).index.astype(str).tolist()


def azure_checkpoint_paths(
    raw_dir: str | Path = DEFAULT_AZURE_RAW_DIR,
    *,
    checkpoint_ids: tuple[int, ...] = AZURE_CHECKPOINT_IDS,
) -> list[Path]:
    """Return the expected local Azure V1 CPU shard paths for a checkpoint curriculum."""
    directory = Path(raw_dir)
    return [
        directory / f"vm_cpu_readings-file-{checkpoint_id}-of-125.csv.gz"
        for checkpoint_id in checkpoint_ids
    ]


def scan_persistent_dense_vms(
    raw_csv_paths: list[str | Path],
    *,
    vm_count: int = 1000,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> list[str]:
    """Find the densest VM IDs present in every Azure CPU shard.

    Each shard is scanned with pandas chunking. Density is the total observation count across the
    persistent intersection, which gives preference to VMs with strong coverage throughout the
    whole six-checkpoint curriculum.
    """
    paths = [Path(path) for path in raw_csv_paths]
    if not paths:
        raise ValueError("raw_csv_paths must include at least one Azure shard")
    if vm_count <= 0:
        raise ValueError("vm_count must be positive")

    persistent_ids: set[str] | None = None
    total_counts: pd.Series | None = None
    for path in paths:
        shard_counts = _count_vm_observations(path, chunksize=chunksize, max_rows=max_rows)
        shard_ids = set(shard_counts.index.astype(str))
        persistent_ids = shard_ids if persistent_ids is None else persistent_ids & shard_ids
        if not persistent_ids:
            raise ValueError("no persistent VM IDs exist across all Azure shards")
        total_counts = shard_counts if total_counts is None else total_counts.add(
            shard_counts, fill_value=0
        )

    if total_counts is None or total_counts.empty:
        raise ValueError("no VM IDs found in Azure CPU shards")
    persistent_counts = total_counts.loc[sorted(persistent_ids)].sort_values(ascending=False)
    return persistent_counts.head(vm_count).index.astype(str).tolist()


def scan_checkpoint_dense_vms(
    raw_csv_paths: list[str | Path],
    *,
    vm_count: int = 1000,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> list[list[str]]:
    """Select each Azure checkpoint shard's own densest VM IDs.

    This intentionally preserves workload composition drift across checkpoints instead of forcing
    the same VM intersection through the whole curriculum.
    """
    paths = [Path(path) for path in raw_csv_paths]
    if not paths:
        raise ValueError("raw_csv_paths must include at least one Azure shard")
    if vm_count <= 0:
        raise ValueError("vm_count must be positive")

    selected_by_checkpoint = []
    for path in paths:
        shard_counts = _count_vm_observations(path, chunksize=chunksize, max_rows=max_rows)
        selected = shard_counts.sort_values(ascending=False).head(vm_count)
        if selected.empty:
            raise ValueError(f"no VM IDs found in {path}")
        selected_by_checkpoint.append(selected.index.astype(str).tolist())
    return selected_by_checkpoint


def load_persistent_vm_matrix(
    raw_csv_path: str | Path,
    *,
    selected_vms: list[str],
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load one Azure CPU shard for a fixed persistent VM set."""
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not selected_vms:
        raise ValueError("selected_vms must contain at least one VM ID")

    selected_set = set(selected_vms)
    frames = []
    rows_seen = 0
    for chunk in _read_headerless_azure_chunks(
        path,
        usecols=["timestamp", "vm_id", "avg_cpu"],
        chunksize=chunksize,
    ):
        rows_seen += len(chunk)
        filtered = chunk[chunk["vm_id"].astype(str).isin(selected_set)].copy()
        if not filtered.empty:
            filtered["vm_id"] = filtered["vm_id"].astype(str)
            filtered["cpu"] = _normalize_cpu(filtered["avg_cpu"])
            frames.append(filtered[["timestamp", "vm_id", "cpu"]])
        if max_rows is not None and rows_seen >= max_rows:
            break

    if not frames:
        raise ValueError(f"no rows matched persistent VM subset in {path}")
    compact = pd.concat(frames, ignore_index=True)
    matrix = compact.pivot_table(
        index="timestamp",
        columns="vm_id",
        values="cpu",
        aggfunc="mean",
    ).sort_index()
    matrix = matrix.reindex(columns=selected_vms)
    matrix = _forward_fill_then_zero(matrix)
    return matrix.clip(lower=0.0, upper=1.0).astype(np.float32)


def load_azure_checkpoint_regimes(
    raw_csv_paths: list[str | Path] | None = None,
    *,
    raw_dir: str | Path = DEFAULT_AZURE_RAW_DIR,
    checkpoint_ids: tuple[int, ...] = AZURE_CHECKPOINT_IDS,
    vm_count: int = 1000,
    selection_strategy: str | VmSelectionStrategy = VmSelectionStrategy.PER_CHECKPOINT_DENSE,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> AzureCheckpointRegimes:
    """Load chronological Azure checkpoint matrices with a configurable VM cohort strategy."""
    paths = [Path(path) for path in raw_csv_paths] if raw_csv_paths else azure_checkpoint_paths(
        raw_dir,
        checkpoint_ids=checkpoint_ids,
    )
    if len(paths) != len(checkpoint_ids):
        raise ValueError("checkpoint_ids must align one-to-one with raw_csv_paths")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    strategy = VmSelectionStrategy(str(selection_strategy))
    if strategy == VmSelectionStrategy.PERSISTENT_DENSE:
        selected_vms = scan_persistent_dense_vms(
            paths,
            vm_count=vm_count,
            chunksize=chunksize,
            max_rows=max_rows,
        )
        selected_by_checkpoint = [selected_vms] * len(paths)
    elif strategy == VmSelectionStrategy.PER_CHECKPOINT_DENSE:
        selected_by_checkpoint = scan_checkpoint_dense_vms(
            paths,
            vm_count=vm_count,
            chunksize=chunksize,
            max_rows=max_rows,
        )
        selected_vms = selected_by_checkpoint[0]
    else:  # pragma: no cover - StrEnum construction validates this path.
        raise AssertionError(f"unhandled VM selection strategy {strategy}")

    matrices = [
        load_persistent_vm_matrix(
            path,
            selected_vms=checkpoint_vms,
            chunksize=chunksize,
            max_rows=max_rows,
        )
        for path, checkpoint_vms in zip(paths, selected_by_checkpoint, strict=True)
    ]
    return AzureCheckpointRegimes(
        checkpoint_ids=list(checkpoint_ids),
        paths=paths,
        selected_vms=selected_vms,
        selected_vms_by_checkpoint=selected_by_checkpoint,
        matrices=matrices,
        selection_strategy=str(strategy),
    )


def chronological_split(matrix: pd.DataFrame) -> ChronologicalSplit:
    """Split a dense CPU matrix into first-half Task A and second-half Task B."""
    if matrix.empty:
        raise ValueError("cannot split an empty matrix")
    midpoint = len(matrix.index) // 2
    if midpoint == 0 or midpoint == len(matrix.index):
        raise ValueError("matrix needs at least two timestamps for chronological split")
    return ChronologicalSplit(
        task_a=matrix.iloc[:midpoint].copy(),
        task_b=matrix.iloc[midpoint:].copy(),
        selected_vms=matrix.columns.astype(str).tolist(),
    )


def detect_azure_columns(columns: pd.Index | list[str]) -> AzureTraceColumns:
    """Detect likely Azure trace columns from common aliases."""
    column_lookup = {_normalize_name(column): str(column) for column in columns}
    resolved: dict[str, str] = {}

    for logical_name, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_name(alias)
            if normalized_alias in column_lookup:
                resolved[logical_name] = column_lookup[normalized_alias]
                break
        if logical_name not in resolved:
            raise ValueError(
                f"could not detect {logical_name!r} column; pass explicit column names in config"
            )

    return AzureTraceColumns(
        timestamp=resolved["timestamp"],
        vm_id=resolved["vm_id"],
        cpu=resolved["cpu"],
    )


def select_vm_subset(frame: pd.DataFrame, *, vm_count: int) -> list[str]:
    """Select VMs deterministically by row coverage, then VM id."""
    coverage = (
        frame.groupby("vm_id", sort=True)["timestamp"]
        .count()
        .rename("row_count")
        .reset_index()
        .sort_values(["row_count", "vm_id"], ascending=[False, True])
    )
    selected = coverage.head(vm_count)["vm_id"].to_numpy()
    if selected.size == 0:
        raise ValueError("no VM ids available in trace")
    return selected.astype(str).tolist()


def align_cpu_matrix(matrix: pd.DataFrame, *, fill_method: str = "interpolate") -> pd.DataFrame:
    """Fill sparse VM readings after timestamp alignment."""
    if matrix.empty:
        raise ValueError("CPU matrix is empty")

    fill_method = fill_method.lower()
    if fill_method == "interpolate":
        aligned = matrix.interpolate(axis=0, limit_direction="both").ffill().bfill()
    elif fill_method == "ffill":
        aligned = matrix.ffill().bfill()
    elif fill_method == "zero":
        aligned = matrix.fillna(0.0)
    elif fill_method in {"none", "preserve"}:
        aligned = matrix
    else:
        raise ValueError("fill_method must be one of: interpolate, ffill, zero, none")

    return aligned.clip(lower=0.0, upper=1.0).astype(np.float32)


def _read_raw_csv(
    raw_csv_path: str | Path,
    *,
    has_header: bool,
    column_names: list[str] | None,
    max_rows: int | None,
) -> pd.DataFrame:
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    read_kwargs: dict[str, Any] = {"nrows": max_rows}
    if has_header:
        read_kwargs["header"] = 0
    else:
        if not column_names:
            raise ValueError("column_names must be supplied when has_header is false")
        read_kwargs["header"] = None
        read_kwargs["names"] = column_names

    return pd.read_csv(path, **read_kwargs)


def _read_raw_csv_chunks(
    raw_csv_path: str | Path,
    *,
    has_header: bool,
    column_names: list[str] | None,
    max_rows: int | None,
    chunksize: int,
):
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    read_kwargs: dict[str, Any] = {"nrows": max_rows, "chunksize": chunksize}
    if has_header:
        read_kwargs["header"] = 0
    else:
        if not column_names:
            raise ValueError("column_names must be supplied when has_header is false")
        read_kwargs["header"] = None
        read_kwargs["names"] = column_names

    yield from pd.read_csv(path, **read_kwargs)


def _read_headerless_azure_chunks(
    raw_csv_path: str | Path,
    *,
    usecols: list[str],
    chunksize: int,
):
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    yield from pd.read_csv(
        path,
        compression="infer",
        header=None,
        names=AZURE_V1_CPU_COLUMNS,
        usecols=usecols,
        chunksize=chunksize,
    )


def _count_vm_observations(
    raw_csv_path: str | Path,
    *,
    chunksize: int,
    max_rows: int | None,
) -> pd.Series:
    counts: pd.Series | None = None
    rows_seen = 0
    for chunk in _read_headerless_azure_chunks(
        raw_csv_path,
        usecols=["vm_id"],
        chunksize=chunksize,
    ):
        rows_seen += len(chunk)
        chunk_counts = chunk["vm_id"].astype(str).value_counts()
        counts = chunk_counts if counts is None else counts.add(chunk_counts, fill_value=0)
        if max_rows is not None and rows_seen >= max_rows:
            break

    if counts is None or counts.empty:
        raise ValueError(f"no VM IDs found in {raw_csv_path}")
    return counts


def _compact_cpu_frame(
    raw: pd.DataFrame,
    *,
    columns: AzureTraceColumns,
    timestamp_unit: str | None,
) -> pd.DataFrame:
    compact = raw[[columns.timestamp, columns.vm_id, columns.cpu]].copy()
    compact.columns = ["timestamp", "vm_id", "cpu"]
    compact = compact.dropna(subset=["timestamp", "vm_id", "cpu"])
    compact["timestamp"] = _coerce_timestamp(compact["timestamp"], unit=timestamp_unit)
    compact["vm_id"] = compact["vm_id"].astype(str)
    compact["cpu"] = _normalize_cpu(compact["cpu"])
    return compact.dropna(subset=["timestamp", "cpu"])


def _forward_fill_then_zero(matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        return matrix.fillna(method="ffill").fillna(0)
    except TypeError:
        return matrix.ffill().fillna(0)


def _normalize_name(name: object) -> str:
    return "".join(character for character in str(name).strip().lower() if character.isalnum())


def _coerce_timestamp(series: pd.Series, *, unit: str | None) -> pd.Series:
    if unit:
        return pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_cpu(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    max_value = numeric.max(skipna=True)
    if pd.notna(max_value) and max_value > 1.5:
        numeric = numeric / 100.0
    return numeric.clip(lower=0.0, upper=1.0)
