"""Materialize a dense subsample from the real local Azure V1 CPU shard.

This script intentionally does not download or synthesize data. It expects the real local file:
``data/raw/vm_cpu_readings-file-1-of-125.csv.gz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from driftscale.traces.azure_loader import (
    DEFAULT_AZURE_V1_CPU_PATH,
    chronological_split,
    load_dense_azure_cpu_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", default=str(DEFAULT_AZURE_V1_CPU_PATH))
    parser.add_argument("--output", default="data/raw/azure_v1_dense_top1000.csv.gz")
    parser.add_argument("--vm-count", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = load_dense_azure_cpu_matrix(
        args.raw_path,
        vm_count=args.vm_count,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    split = chronological_split(matrix)
    matrix.to_csv(output, compression="infer")
    if not args.quiet:
        print(f"Saved dense Azure matrix to {output}")
        print(f"Matrix shape: {matrix.shape[0]} timestamps x {matrix.shape[1]} VMs")
        print(f"Task A timestamps: {len(split.task_a)}")
        print(f"Task B timestamps: {len(split.task_b)}")


if __name__ == "__main__":
    main()

