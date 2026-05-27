import gzip

import numpy as np
import pandas as pd

from driftscale.traces.azure_loader import (
    AzureTraceColumns,
    chronological_split,
    load_azure_checkpoint_regimes,
    load_azure_cpu_matrix,
    load_dense_azure_cpu_matrix,
    scan_persistent_dense_vms,
)
from driftscale.traces.preprocess import DemandMappingConfig, build_preprocessed_trace


def test_azure_loader_filters_vms_normalizes_cpu_and_aligns_timestamps(tmp_path) -> None:
    raw_csv = tmp_path / "azure_sample.csv"
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2017-01-01T00:00:00Z",
                "2017-01-01T00:00:00Z",
                "2017-01-01T00:00:00Z",
                "2017-01-01T00:05:00Z",
                "2017-01-01T00:05:00Z",
                "2017-01-01T00:10:00Z",
                "2017-01-01T00:10:00Z",
            ],
            "vm_id": ["vm-a", "vm-b", "vm-c", "vm-a", "vm-c", "vm-a", "vm-c"],
            "avg_cpu": [10, 20, 30, 40, 95, 20, 35],
        }
    )
    frame.to_csv(raw_csv, index=False)

    matrix = load_azure_cpu_matrix(
        raw_csv,
        vm_count=2,
        columns=AzureTraceColumns(timestamp="timestamp", vm_id="vm_id", cpu="avg_cpu"),
    )

    assert matrix.shape == (3, 2)
    assert matrix.columns.tolist() == ["vm-a", "vm-c"]
    assert np.isclose(matrix.loc[pd.Timestamp("2017-01-01T00:05:00Z"), "vm-c"], 0.95)
    assert matrix.max().max() <= 1.0
    assert not matrix.isna().any().any()


def test_build_preprocessed_trace_includes_configurable_threshold_alpha() -> None:
    matrix = pd.DataFrame(
        [[0.2, 0.3], [0.95, 0.4]],
        index=pd.date_range("2017-01-01", periods=2, freq="5min", tz="UTC"),
    )

    trace = build_preprocessed_trace(
        matrix,
        mapping_config=DemandMappingConfig(variant="threshold", scale_factor=1.0, alpha=4.0),
    )

    np.testing.assert_allclose(trace["demand"].to_numpy(), np.array([0.5, 5.35]))
    assert trace["alpha"].tolist() == [4.0, 4.0]


def test_dense_headerless_azure_loader_selects_top_vms_and_splits(tmp_path) -> None:
    raw_gzip = tmp_path / "vm_cpu_readings.csv.gz"
    rows = [
        "0,vm-a,0,10,10",
        "0,vm-b,0,20,20",
        "0,vm-c,0,30,30",
        "1,vm-a,0,40,40",
        "1,vm-c,0,50,50",
        "2,vm-a,0,60,60",
        "2,vm-b,0,70,70",
        "2,vm-c,0,80,80",
        "3,vm-a,0,90,90",
        "3,vm-c,0,100,100",
    ]
    with gzip.open(raw_gzip, "wt", encoding="utf-8") as file:
        file.write("\n".join(rows))

    matrix = load_dense_azure_cpu_matrix(raw_gzip, vm_count=2, chunksize=3)
    split = chronological_split(matrix)

    assert matrix.shape == (4, 2)
    assert matrix.columns.tolist() == ["vm-a", "vm-c"]
    assert np.isclose(matrix.loc[0, "vm-a"], 0.10)
    assert np.isclose(matrix.loc[3, "vm-c"], 1.0)
    assert len(split.task_a) == 2
    assert len(split.task_b) == 2


def test_checkpoint_loader_intersects_persistent_vms_and_preserves_regime_order(
    tmp_path,
) -> None:
    raw_files = [
        _write_headerless_azure_gzip(
            tmp_path / "vm_cpu_readings-file-1-of-125.csv.gz",
            [
                "0,vm-a,0,10,10",
                "0,vm-b,0,20,20",
                "0,vm-c,0,30,30",
                "1,vm-a,0,40,40",
                "2,vm-a,0,50,50",
            ],
        ),
        _write_headerless_azure_gzip(
            tmp_path / "vm_cpu_readings-file-25-of-125.csv.gz",
            [
                "0,vm-a,0,11,11",
                "0,vm-b,0,21,21",
                "1,vm-a,0,41,41",
                "1,vm-c,0,31,31",
            ],
        ),
        _write_headerless_azure_gzip(
            tmp_path / "vm_cpu_readings-file-50-of-125.csv.gz",
            [
                "0,vm-a,0,12,12",
                "0,vm-b,0,22,22",
                "1,vm-a,0,42,42",
                "1,vm-c,0,32,32",
            ],
        ),
    ]

    selected = scan_persistent_dense_vms(raw_files, vm_count=2, chunksize=2)
    regimes = load_azure_checkpoint_regimes(
        raw_files,
        checkpoint_ids=(1, 25, 50),
        vm_count=2,
        chunksize=2,
    )

    assert selected == ["vm-a", "vm-b"]
    assert regimes.checkpoint_ids == [1, 25, 50]
    assert regimes.selected_vms == ["vm-a", "vm-b"]
    assert [matrix.shape for matrix in regimes.matrices] == [(3, 2), (2, 2), (2, 2)]
    assert np.isclose(regimes.matrices[1].loc[1, "vm-b"], 0.21)


def _write_headerless_azure_gzip(path, rows: list[str]):
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("\n".join(rows))
    return path
