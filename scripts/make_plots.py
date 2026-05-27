"""Generate DriftScale evaluation plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", default="results/calibration/baseline_metrics.csv")
    parser.add_argument("--ppo-metrics", default="results/ppo_vanilla/metrics.csv")
    parser.add_argument("--output", default="media/cost_vs_slo.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_metrics = pd.read_csv(args.baseline_metrics)
    ppo_metrics = pd.read_csv(args.ppo_metrics)
    metrics = pd.concat([baseline_metrics, ppo_metrics], ignore_index=True)
    if "total_cost" not in metrics.columns:
        metrics["total_cost"] = metrics["total_resource_cost"] + metrics["total_scaling_cost"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    label_offsets = [(8, 8), (8, -16), (-86, 8), (-92, -18), (14, 20), (-90, 22)]
    for index, row in metrics.iterrows():
        slo_pct = float(row["slo_violation_rate"]) * 100.0
        cost = float(row["total_cost"])
        label = str(row["policy"])
        plt.scatter(slo_pct, cost, s=90)
        plt.annotate(
            label,
            (slo_pct, cost),
            textcoords="offset points",
            xytext=label_offsets[index % len(label_offsets)],
            fontsize=9,
            arrowprops={"arrowstyle": "-", "alpha": 0.25, "linewidth": 0.8},
        )

    frontier = metrics.sort_values("slo_violation_rate")
    plt.plot(
        frontier["slo_violation_rate"] * 100.0,
        frontier["total_cost"],
        linewidth=1,
        alpha=0.35,
    )
    plt.xlabel("SLO violation rate (%)")
    plt.ylabel("Total simulated cost")
    plt.title("Cost vs. SLO Pareto Frontier")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
