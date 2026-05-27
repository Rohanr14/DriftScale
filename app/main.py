"""Tiny FastAPI target service for the ECS Fargate validation demo."""

from __future__ import annotations

import time

from fastapi import FastAPI, Query

app = FastAPI(title="DriftScale Demo Target")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cpu")
def cpu(work: int = Query(default=50, ge=0, le=5_000)) -> dict[str, float | int]:
    """Burn CPU for roughly ``work`` milliseconds and return timing metadata."""
    start = time.perf_counter()
    deadline = start + (work / 1_000.0)
    iterations = 0
    accumulator = 0

    while time.perf_counter() < deadline:
        iterations += 1
        accumulator = (accumulator + (iterations * 17)) % 1_000_003

    elapsed_ms = (time.perf_counter() - start) * 1_000.0
    return {
        "requested_work_ms": work,
        "elapsed_ms": round(elapsed_ms, 3),
        "iterations": iterations,
        "checksum": accumulator,
    }
