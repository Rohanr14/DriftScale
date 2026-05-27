"""Smoke tests for the replay-ratio ablation summary/plot logic.

These do not run PPO — they fabricate per-seed BWT inputs to exercise the bootstrap
summary and plot writer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_replay_ratio_ablation import (  # noqa: E402
    summarize_ablation,
    summary_markdown_table,
    write_ablation_plot,
)


def _fake_raw(mix_ratios: list[float], seeds: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ratio in mix_ratios:
        for seed in seeds:
            rows.append(
                {
                    "mapping": "linear",
                    "mix_ratio": ratio,
                    "seed": seed,
                    "final_task_1_bwt": float(rng.normal(loc=100.0 * ratio, scale=20.0)),
                    "final_mean_prior_bwt": float(rng.normal(loc=80.0 * ratio, scale=15.0)),
                }
            )
    return pd.DataFrame(rows)


def test_summarize_ablation_emits_one_row_per_ratio() -> None:
    raw = _fake_raw([0.0, 0.25, 0.5, 0.75], [7, 8, 9, 10, 11])
    summary = summarize_ablation(raw, bootstrap_resamples=500, alpha=0.05)
    assert list(summary["mix_ratio"]) == [0.0, 0.25, 0.5, 0.75]
    for column in (
        "task_1_bwt_mean",
        "task_1_bwt_ci_low",
        "task_1_bwt_ci_high",
        "mean_prior_bwt_mean",
    ):
        assert column in summary.columns
    for _, row in summary.iterrows():
        assert row["task_1_bwt_ci_low"] <= row["task_1_bwt_mean"] <= row["task_1_bwt_ci_high"]


def test_summary_markdown_includes_one_row_per_ratio() -> None:
    raw = _fake_raw([0.0, 0.5], [7, 8, 9])
    summary = summarize_ablation(raw, bootstrap_resamples=200, alpha=0.05)
    markdown = summary_markdown_table(summary)
    # Header + separator + one row per ratio.
    assert markdown.count("\n") == 1 + 1 + 1


def test_write_ablation_plot_produces_a_file(tmp_path) -> None:
    raw = _fake_raw([0.0, 0.25, 0.5], [7, 8, 9])
    summary = summarize_ablation(raw, bootstrap_resamples=200, alpha=0.05)
    output = tmp_path / "replay_ratio_ablation.png"
    write_ablation_plot(summary, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0
