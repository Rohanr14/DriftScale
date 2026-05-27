"""Train the vanilla MaskablePPO autoscaling baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from driftscale.envs.cost_env import DriftScaleEnv
from driftscale.envs.reward import RewardConfig
from driftscale.traces.preprocess import load_preprocessed_demand
from driftscale.utils.config import load_yaml
from driftscale.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/ppo.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    metrics = train_maskable_ppo(config)
    print(f"Saved model to {metrics['model_path']}")
    print(f"Saved VecNormalize stats to {metrics['vecnormalize_path']}")
    print(f"Saved metrics to {metrics['metrics_path']}")
    print(
        "PPO SLO violation rate: "
        f"{metrics['slo_violation_rate']:.3f} "
        f"({metrics['slo_violation_count']}/{metrics['steps']} steps)"
    )


def train_maskable_ppo(config: dict[str, Any]) -> dict[str, float | int | str]:
    """Train or fine-tune a MaskablePPO policy from a config dictionary."""
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    output_dir = Path(config.get("output_dir", "results/ppo_vanilla"))
    output_dir.mkdir(parents=True, exist_ok=True)

    demand = load_preprocessed_demand(
        config["trace"]["path"],
        demand_column=config["trace"].get("demand_column", "demand"),
    )
    env = build_vecnormalize_env(
        config,
        demand=demand,
        seed=seed,
        vecnormalize_path=config.get("init_vecnormalize_path"),
        training=True,
    )
    init_model_path = config.get("init_model_path")
    if init_model_path:
        model = MaskablePPO.load(init_model_path, env=env, seed=seed)
    else:
        model = build_model(config, env=env, seed=seed)
    model.learn(
        total_timesteps=int(config["ppo"].get("total_timesteps", 2048)),
        reset_num_timesteps=not bool(init_model_path),
    )

    model_path = output_dir / "model.zip"
    vecnormalize_path = output_dir / "vecnormalize.pkl"
    model.save(model_path)
    env.save(vecnormalize_path)

    env.training = False
    env.norm_reward = False
    metrics = evaluate_maskable_ppo(
        model,
        env,
        deterministic=bool(config.get("eval", {}).get("deterministic", False)),
    )
    metrics["policy"] = "ppo_vanilla"
    metrics["model_path"] = str(model_path)
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    metrics["vecnormalize_path"] = str(vecnormalize_path)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def build_vecnormalize_env(
    config: dict[str, Any],
    *,
    demand: np.ndarray,
    seed: int,
    n_envs: int = 1,
    vecnormalize_path: str | Path | None = None,
    training: bool = True,
) -> VecNormalize:
    """Create DriftScaleEnv wrapped exactly as required for Week 4 PPO training."""
    vec_cfg = config.get("vecnormalize", {})
    vec_env = build_dummy_vec_env(config, demands=[demand] * n_envs, seed=seed)

    if vecnormalize_path:
        env = VecNormalize.load(str(vecnormalize_path), vec_env)
        env.training = training
        env.norm_reward = bool(vec_cfg.get("norm_reward", True)) if training else False
        return env

    return VecNormalize(
        vec_env,
        norm_obs=bool(vec_cfg.get("norm_obs", True)),
        norm_reward=bool(vec_cfg.get("norm_reward", True)),
        clip_obs=float(vec_cfg.get("clip_obs", 10.0)),
        clip_reward=float(vec_cfg.get("clip_reward", 100.0)),
        gamma=float(config.get("ppo", {}).get("gamma", 0.99)),
        training=training,
    )


def build_dummy_vec_env(
    config: dict[str, Any],
    *,
    demands: list[np.ndarray],
    seed: int,
) -> DummyVecEnv:
    """Build a vectorized set of DriftScaleEnv instances."""
    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))

    def make_env(demand: np.ndarray, rank: int):
        def _init() -> DriftScaleEnv:
            env = DriftScaleEnv(
                demand,
                min_tasks=int(env_cfg.get("min_tasks", 1)),
                max_tasks=int(env_cfg.get("max_tasks", 20)),
                initial_tasks=int(env_cfg.get("initial_tasks", 4)),
                capacity_per_task=float(env_cfg.get("capacity_per_task", 1.0)),
                reward_config=reward_cfg,
                step_minutes=int(env_cfg.get("step_minutes", 5)),
                strict_action_mask=True,
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([make_env(demand, rank) for rank, demand in enumerate(demands)])


def build_model(config: dict[str, Any], *, env: VecNormalize, seed: int) -> MaskablePPO:
    ppo_cfg = config.get("ppo", {})
    policy_kwargs = config.get("policy_kwargs", {})
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 64)),
        batch_size=int(ppo_cfg.get("batch_size", 32)),
        n_epochs=int(ppo_cfg.get("n_epochs", 5)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        seed=seed,
        verbose=int(ppo_cfg.get("verbose", 0)),
        policy_kwargs=policy_kwargs,
    )


def evaluate_maskable_ppo(
    model: MaskablePPO,
    env: VecEnv,
    *,
    deterministic: bool,
) -> dict[str, float | int]:
    obs = env.reset()
    done = np.array([False])
    total_reward = 0.0
    resource_cost = 0.0
    scaling_cost = 0.0
    slo_violations = 0
    scale_actions = 0
    task_counts: list[int] = []
    overprovision: list[float] = []
    steps = 0

    while not bool(done[0]):
        masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        obs, rewards, done, infos = env.step(action)
        info = infos[0]
        total_reward += float(rewards[0])
        resource_cost += float(info["resource_cost"])
        scaling_cost += float(info["scaling_action_penalty"])
        slo_violations += int(info["slo_violation"])
        scale_actions += int(info["action_delta"] != 0)
        task_counts.append(int(info["task_count"]))
        overprovision.append(float(info["overprovision_ratio"]))
        steps += 1

    return {
        "total_reward": total_reward,
        "total_resource_cost": resource_cost,
        "total_scaling_cost": scaling_cost,
        "total_cost": resource_cost + scaling_cost,
        "slo_violation_rate": slo_violations / max(steps, 1),
        "slo_violation_count": slo_violations,
        "scale_action_count": scale_actions,
        "mean_task_count": float(np.mean(task_counts)),
        "mean_overprovision_ratio": float(np.mean(overprovision)),
        "final_task_count": int(task_counts[-1]),
        "steps": steps,
    }


def evaluate_saved_model(
    *,
    model_path: str | Path,
    vecnormalize_path: str | Path,
    config: dict[str, Any],
    demand: np.ndarray,
    deterministic: bool = True,
) -> dict[str, float | int]:
    """Load a saved MaskablePPO model and evaluate it on one demand episode."""
    seed = int(config.get("seed", 0))
    env = build_vecnormalize_env(
        config,
        demand=demand,
        seed=seed,
        vecnormalize_path=vecnormalize_path,
        training=False,
    )
    env.norm_reward = False
    model = MaskablePPO.load(str(model_path), env=env)
    return evaluate_maskable_ppo(model, env, deterministic=deterministic)


if __name__ == "__main__":
    main()
