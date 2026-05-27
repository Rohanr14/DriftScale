"""Preprocess Azure VM CPU traces into lightweight DriftScale demand caches."""

from __future__ import annotations

import argparse

from driftscale.traces.azure_loader import AzureTraceColumns, load_azure_cpu_matrix
from driftscale.traces.preprocess import (
    DemandMappingConfig,
    build_preprocessed_trace,
    write_preprocessed_trace,
)
from driftscale.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/azure_v1.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    loader_cfg = config.get("loader", {})
    column_cfg = loader_cfg.get("columns", {})
    columns = None
    if column_cfg:
        columns = AzureTraceColumns(
            timestamp=column_cfg.get("timestamp", "timestamp"),
            vm_id=column_cfg.get("vm_id", "vm_id"),
            cpu=column_cfg.get("cpu", "avg_cpu"),
        )

    cpu_matrix = load_azure_cpu_matrix(
        config["raw_csv_path"],
        vm_count=int(loader_cfg.get("vm_count", 500)),
        columns=columns,
        has_header=bool(loader_cfg.get("has_header", True)),
        column_names=loader_cfg.get("column_names"),
        max_rows=loader_cfg.get("max_rows"),
        chunksize=loader_cfg.get("chunksize"),
        fill_method=str(loader_cfg.get("fill_method", "interpolate")),
        timestamp_unit=loader_cfg.get("timestamp_unit"),
    )
    mapping_config = DemandMappingConfig(**config.get("mapping", {}))
    preprocessed = build_preprocessed_trace(cpu_matrix, mapping_config=mapping_config)
    output_path = write_preprocessed_trace(preprocessed, config["output_path"])

    print(f"Loaded CPU matrix: {cpu_matrix.shape[0]} timesteps x {cpu_matrix.shape[1]} VMs")
    print(f"Mapping variant: {mapping_config.variant}")
    print(f"Scale factor: {mapping_config.scale_factor}")
    print(f"Threshold alpha: {mapping_config.alpha}")
    print(f"Demand range: {preprocessed['demand'].min():.3f} - {preprocessed['demand'].max():.3f}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
