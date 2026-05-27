"""Train MaskablePPO with rollout-level interleaved replay.

This intentionally avoids an off-policy replay buffer. PPO remains on-policy because old and new
regimes are mixed as live vectorized environments during rollout collection.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from driftscale.agents.train_ppo import (
    build_dummy_vec_env,
    build_model,
    build_vecnormalize_env,
    evaluate_maskable_ppo,
)
from driftscale.traces.preprocess import load_preprocessed_demand
from driftscale.utils.config import load_yaml
from driftscale.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/ppo_replay.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    metrics = train_replay_ppo(config)
    print(f"Replay mix ratio: {metrics['replay_mix_ratio']:.2f}")
    print(f"Vectorized envs old/new: {metrics['previous_envs']}/{metrics['current_envs']}")
    print(f"Saved model to {metrics['model_path']}")
    print(
        "Replay Task-A SLO violation rate: "
        f"{metrics['slo_violation_rate']:.3f} "
        f"({metrics['slo_violation_count']}/{metrics['steps']} steps)"
    )


def train_replay_ppo(config: dict[str, Any]) -> dict[str, float | int | str]:
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    current_task = _current_task_config(config)
    previous_tasks = _previous_task_configs(config)
    current_demand = _load_task_demand(current_task)
    previous_demands = [_load_task_demand(task) for task in previous_tasks]
    eval_task = config.get("eval_task") or (previous_tasks[0] if previous_tasks else current_task)
    eval_demand = _load_task_demand(eval_task)

    replay_cfg = config.get("replay", {})
    n_envs = int(replay_cfg.get("n_envs", 4))
    replay_mix_ratio = float(replay_cfg.get("replay_mix_ratio", 0.25))
    if not 0.0 <= replay_mix_ratio < 1.0:
        raise ValueError("replay_mix_ratio must satisfy 0.0 <= ratio < 1.0")
    current_envs, previous_envs = _allocate_replay_envs(
        n_envs=n_envs,
        replay_mix_ratio=replay_mix_ratio,
        previous_task_count=len(previous_demands),
    )
    demands = [current_demand] * current_envs + _round_robin_demands(
        previous_demands,
        env_count=previous_envs,
    )
    raw_vec_env = build_dummy_vec_env(config, demands=demands, seed=seed)

    init_vecnormalize_path = config.get("init_vecnormalize_path")
    if init_vecnormalize_path:
        env = VecNormalize.load(str(init_vecnormalize_path), raw_vec_env)
        env.training = True
        env.norm_reward = bool(config.get("vecnormalize", {}).get("norm_reward", True))
    else:
        vec_cfg = config.get("vecnormalize", {})
        env = VecNormalize(
            raw_vec_env,
            norm_obs=bool(vec_cfg.get("norm_obs", True)),
            norm_reward=bool(vec_cfg.get("norm_reward", True)),
            clip_obs=float(vec_cfg.get("clip_obs", 10.0)),
            clip_reward=float(vec_cfg.get("clip_reward", 100.0)),
            gamma=float(config.get("ppo", {}).get("gamma", 0.99)),
        )

    init_model_path = config.get("init_model_path")
    if init_model_path:
        model = MaskablePPO.load(str(init_model_path), env=env, seed=seed)
    else:
        model = build_model(config, env=env, seed=seed)
    model.learn(
        total_timesteps=int(config["ppo"].get("total_timesteps", 2048)),
        reset_num_timesteps=not bool(init_model_path),
    )

    output_dir = Path(config.get("output_dir", "results/ppo_replay"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.zip"
    vecnormalize_path = output_dir / "vecnormalize.pkl"
    metrics_path = output_dir / "metrics.csv"
    model.save(model_path)
    env.save(vecnormalize_path)

    eval_env = build_vecnormalize_env(
        config,
        demand=eval_demand,
        seed=seed,
        vecnormalize_path=vecnormalize_path,
        training=False,
    )
    eval_env.norm_reward = False
    metrics = evaluate_maskable_ppo(
        model,
        eval_env,
        deterministic=bool(config.get("eval", {}).get("deterministic", True)),
    )
    metrics["policy"] = "ppo_replay"
    metrics["model_path"] = str(model_path)
    metrics["vecnormalize_path"] = str(vecnormalize_path)
    metrics["metrics_path"] = str(metrics_path)
    metrics["replay_mix_ratio"] = replay_mix_ratio
    metrics["previous_envs"] = previous_envs
    metrics["current_envs"] = current_envs
    metrics["effective_replay_mix_ratio"] = previous_envs / max(previous_envs + current_envs, 1)
    metrics["previous_task_count"] = len(previous_demands)
    metrics["task_a_envs"] = previous_envs
    metrics["task_b_envs"] = current_envs
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    return metrics


def _current_task_config(config: dict[str, Any]) -> dict[str, Any]:
    if "current_task" in config:
        return config["current_task"]
    if "task_b" in config:
        return config["task_b"]
    raise KeyError("replay config must include current_task or legacy task_b")


def _previous_task_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "previous_tasks" in config:
        return list(config["previous_tasks"])
    if "task_a" in config:
        return [config["task_a"]]
    return []


def _load_task_demand(task_config: dict[str, Any]) -> Any:
    return load_preprocessed_demand(
        task_config["path"],
        demand_column=task_config.get("demand_column", "demand"),
    )


def _allocate_replay_envs(
    *,
    n_envs: int,
    replay_mix_ratio: float,
    previous_task_count: int,
) -> tuple[int, int]:
    if n_envs <= 0:
        raise ValueError("replay n_envs must be positive")
    if previous_task_count <= 0 or replay_mix_ratio == 0.0:
        return n_envs, 0

    requested_previous_envs = max(1, math.ceil(n_envs * replay_mix_ratio))
    previous_envs = max(requested_previous_envs, previous_task_count)
    current_envs = max(1, n_envs - previous_envs)
    return current_envs, previous_envs


def _round_robin_demands(demands: list[Any], *, env_count: int) -> list[Any]:
    if env_count <= 0:
        return []
    if not demands:
        raise ValueError("cannot allocate replay envs without previous task demands")
    return [demands[index % len(demands)] for index in range(env_count)]


if __name__ == "__main__":
    main()
