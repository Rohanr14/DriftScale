"""Run abrupt-drift forgetting and replay experiments."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from driftscale.agents.train_ppo import evaluate_saved_model, train_maskable_ppo
from driftscale.agents.train_ppo_replay import train_replay_ppo
from driftscale.eval.forgetting import ForgettingMetrics
from driftscale.traces.azure_loader import AzureTraceColumns, load_azure_cpu_matrix
from driftscale.traces.preprocess import (
    DemandMappingConfig,
    build_preprocessed_trace,
    write_preprocessed_trace,
)
from driftscale.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-a-env-config", default="configs/env/azure_v1.yaml")
    parser.add_argument("--task-b-env-config", default="configs/env/azure_v2.yaml")
    parser.add_argument("--ppo-config", default="configs/train/ppo.yaml")
    parser.add_argument("--replay-config", default="configs/train/ppo_replay.yaml")
    parser.add_argument("--output-dir", default="results/drift_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_config = load_yaml(args.ppo_config)
    replay_config = load_yaml(args.replay_config)
    scale_factor = float(ppo_config["calibration"]["scale_factor"])
    capacity_per_task = float(ppo_config["calibration"]["capacity_per_task"])

    task_a_linear = preprocess_from_env_config(Path(args.task_a_env_config))
    task_b_linear = preprocess_from_env_config(Path(args.task_b_env_config))
    task_a_path = write_calibrated_trace(
        task_a_linear,
        output_dir / "azure_v1_task_a.csv",
        scale_factor=scale_factor,
        capacity_per_task=capacity_per_task,
    )
    task_b_path = write_calibrated_trace(
        task_b_linear,
        output_dir / "azure_v2_task_b.csv",
        scale_factor=scale_factor,
        capacity_per_task=capacity_per_task,
    )

    task_a_config = build_single_task_config(
        ppo_config,
        trace_path=task_a_path,
        output_dir=output_dir / "task_a_model",
        capacity_per_task=capacity_per_task,
    )
    task_a_metrics = train_maskable_ppo(task_a_config)
    task_a_eval = evaluate_task_a(
        task_a_config,
        task_a_path=task_a_path,
        model_path=task_a_metrics["model_path"],
        vecnormalize_path=task_a_metrics["vecnormalize_path"],
    )

    naive_config = build_single_task_config(
        ppo_config,
        trace_path=task_b_path,
        output_dir=output_dir / "naive_finetune_b",
        capacity_per_task=capacity_per_task,
        init_model_path=task_a_metrics["model_path"],
        init_vecnormalize_path=task_a_metrics["vecnormalize_path"],
    )
    naive_metrics = train_maskable_ppo(naive_config)
    naive_task_a_eval = evaluate_task_a(
        naive_config,
        task_a_path=task_a_path,
        model_path=naive_metrics["model_path"],
        vecnormalize_path=naive_metrics["vecnormalize_path"],
    )

    replay_train_config = build_replay_config(
        replay_config,
        task_a_path=task_a_path,
        task_b_path=task_b_path,
        output_dir=output_dir / "replay_finetune_b",
        capacity_per_task=capacity_per_task,
        init_model_path=task_a_metrics["model_path"],
        init_vecnormalize_path=task_a_metrics["vecnormalize_path"],
    )
    replay_metrics = train_replay_ppo(replay_train_config)
    replay_task_a_eval = evaluate_task_a(
        replay_train_config,
        task_a_path=task_a_path,
        model_path=replay_metrics["model_path"],
        vecnormalize_path=replay_metrics["vecnormalize_path"],
    )

    forgetting_rows = [
        ForgettingMetrics(
            method="naive_finetuning",
            task_a_reward_before_b=float(task_a_eval["total_reward"]),
            task_a_reward_after_b=float(naive_task_a_eval["total_reward"]),
        ).as_dict(),
        ForgettingMetrics(
            method="ppo_replay",
            task_a_reward_before_b=float(task_a_eval["total_reward"]),
            task_a_reward_after_b=float(replay_task_a_eval["total_reward"]),
        ).as_dict(),
    ]
    forgetting = pd.DataFrame(forgetting_rows)
    forgetting["task_a_slo_after_b"] = [
        naive_task_a_eval["slo_violation_rate"],
        replay_task_a_eval["slo_violation_rate"],
    ]
    forgetting["task_a_scale_actions_after_b"] = [
        naive_task_a_eval["scale_action_count"],
        replay_task_a_eval["scale_action_count"],
    ]
    forgetting_path = output_dir / "forgetting.csv"
    forgetting.to_csv(forgetting_path, index=False)

    print("\nAbrupt Drift Forgetting Evaluation")
    print(f"Task A trace: {task_a_path}")
    print(f"Task B trace: {task_b_path}")
    print(f"Replay mix ratio: {replay_train_config['replay']['replay_mix_ratio']:.2f}")
    print(
        forgetting[
            [
                "method",
                "task_a_reward_before_b",
                "task_a_reward_after_b",
                "bwt",
                "forgetting",
                "task_a_slo_after_b",
            ]
        ].to_string(index=False)
    )
    print(f"Saved forgetting metrics to {forgetting_path}")


def preprocess_from_env_config(config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
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
    preprocessed = build_preprocessed_trace(
        cpu_matrix,
        mapping_config=DemandMappingConfig(**config.get("mapping", {})),
    )
    write_preprocessed_trace(preprocessed, config["output_path"])
    return preprocessed


def write_calibrated_trace(
    trace: pd.DataFrame,
    output_path: Path,
    *,
    scale_factor: float,
    capacity_per_task: float,
) -> Path:
    calibrated = trace.copy()
    calibrated["demand"] = calibrated["demand"].astype("float32") * scale_factor
    calibrated["scale_factor"] = scale_factor
    calibrated["capacity_per_task"] = capacity_per_task
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_path, index=False)
    return output_path


def build_single_task_config(
    base_config: dict,
    *,
    trace_path: Path,
    output_dir: Path,
    capacity_per_task: float,
    init_model_path: str | Path | None = None,
    init_vecnormalize_path: str | Path | None = None,
) -> dict:
    config = copy.deepcopy(base_config)
    config["trace"] = {"path": str(trace_path), "demand_column": "demand"}
    config["output_dir"] = str(output_dir)
    config.setdefault("env", {})["capacity_per_task"] = capacity_per_task
    config.setdefault("env", {}).setdefault("initial_tasks", 1)
    config.setdefault("eval", {})["deterministic"] = True
    if init_model_path:
        config["init_model_path"] = str(init_model_path)
    else:
        config.pop("init_model_path", None)
    if init_vecnormalize_path:
        config["init_vecnormalize_path"] = str(init_vecnormalize_path)
    else:
        config.pop("init_vecnormalize_path", None)
    return config


def build_replay_config(
    base_config: dict,
    *,
    task_a_path: Path,
    task_b_path: Path,
    output_dir: Path,
    capacity_per_task: float,
    init_model_path: str | Path,
    init_vecnormalize_path: str | Path,
) -> dict:
    config = copy.deepcopy(base_config)
    config["task_a"] = {"path": str(task_a_path), "demand_column": "demand"}
    config["task_b"] = {"path": str(task_b_path), "demand_column": "demand"}
    config["output_dir"] = str(output_dir)
    config["init_model_path"] = str(init_model_path)
    config["init_vecnormalize_path"] = str(init_vecnormalize_path)
    config.setdefault("env", {})["capacity_per_task"] = capacity_per_task
    config.setdefault("env", {}).setdefault("initial_tasks", 1)
    config.setdefault("eval", {})["deterministic"] = True
    return config


def evaluate_task_a(
    config: dict,
    *,
    task_a_path: Path,
    model_path: str | Path,
    vecnormalize_path: str | Path,
) -> dict[str, float | int]:
    eval_config = copy.deepcopy(config)
    eval_config["trace"] = {"path": str(task_a_path), "demand_column": "demand"}
    demand = pd.read_csv(task_a_path)["demand"].to_numpy(dtype="float32")
    return evaluate_saved_model(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        config=eval_config,
        demand=demand,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
