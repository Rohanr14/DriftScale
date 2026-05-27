"""Safety-wrapped AWS ECS Fargate control loop for the DriftScale demo."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from sb3_contrib import MaskablePPO

from driftscale.agents.train_ppo import build_vecnormalize_env
from driftscale.aws.safety import EcsSafetyWrapper, SafetyConfig
from driftscale.envs.spaces import ACTION_DELTAS, action_mask, normalize_action_delta
from driftscale.utils.config import load_yaml


@dataclass(frozen=True)
class ControllerConfig:
    region: str
    cluster_name: str
    service_name: str
    alb_arn_suffix: str
    target_group_arn_suffix: str | None
    model_path: Path
    vecnormalize_path: Path
    deterministic: bool
    observation_min_tasks: int
    observation_max_tasks: int
    safety: SafetyConfig
    interval_seconds: int
    max_iterations: int | None
    metric_period_seconds: int
    metric_lookback_minutes: int
    latency_slo_seconds: float


class ObservationBuilder:
    """Build DriftScale's 12-feature observation from live CloudWatch readings."""

    def __init__(
        self,
        *,
        safety: SafetyConfig,
        observation_min_tasks: int,
        observation_max_tasks: int,
        latency_slo_seconds: float,
    ) -> None:
        self.safety = safety
        self.observation_min_tasks = observation_min_tasks
        self.observation_max_tasks = observation_max_tasks
        self.latency_slo_seconds = latency_slo_seconds
        self.cpu_history: deque[float] = deque([0.0] * 60, maxlen=60)
        self.slo_history: deque[float] = deque([0.0] * 15, maxlen=15)
        self.scale_history: deque[float] = deque([0.0] * 15, maxlen=15)
        self.previous_delta = 0

    def build(
        self,
        *,
        cpu_utilization_percent: float,
        target_response_time_seconds: float,
        desired_count: int,
        now: datetime,
    ) -> np.ndarray:
        cpu = float(np.clip(cpu_utilization_percent / 100.0, 0.0, 1.0))
        self.cpu_history.append(cpu)
        self.slo_history.append(float(target_response_time_seconds > self.latency_slo_seconds))

        cpu_values = list(self.cpu_history)
        step_of_day = (
            (now.hour * 60 * 60 + now.minute * 60 + now.second) / float(24 * 60 * 60)
        )
        angle = 2.0 * math.pi * step_of_day

        return np.asarray(
            [
                cpu,
                float(np.mean(cpu_values[-5:])),
                float(np.mean(cpu_values[-15:])),
                float(np.mean(cpu_values[-60:])),
                float(np.max(cpu_values[-15:])),
                float(np.std(cpu_values[-15:])),
                normalize_task_count(
                    desired_count,
                    self.observation_min_tasks,
                    self.observation_max_tasks,
                ),
                normalize_action_delta(self.previous_delta),
                float(math.sin(angle)),
                float(math.cos(angle)),
                float(np.mean(self.slo_history)),
                float(np.mean(self.scale_history)),
            ],
            dtype=np.float32,
        )

    def record_scale_action(self, applied_delta: int) -> None:
        self.previous_delta = applied_delta
        self.scale_history.append(float(applied_delta != 0))


class DriftScaleAwsController:
    """Read AWS metrics, ask the trained policy for an action, and safely scale ECS."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        session = boto3.Session(region_name=config.region)
        self.ecs = session.client("ecs")
        self.cloudwatch = session.client("cloudwatch")
        self.model = MaskablePPO.load(str(config.model_path))
        self.vecnormalize = build_vecnormalize_env(
            {
                "env": {
                    "min_tasks": config.observation_min_tasks,
                    "max_tasks": config.observation_max_tasks,
                    "initial_tasks": config.observation_min_tasks,
                    "capacity_per_task": 1.0,
                },
                "vecnormalize": {
                    "norm_obs": True,
                    "norm_reward": True,
                    "clip_obs": 10.0,
                    "clip_reward": 100.0,
                },
                "ppo": {"gamma": 0.99},
            },
            demand=np.asarray([1.0, 1.0], dtype=np.float32),
            seed=0,
            vecnormalize_path=config.vecnormalize_path,
            training=False,
        )
        self.vecnormalize.norm_reward = False
        self.safety = EcsSafetyWrapper(config.safety)
        self.observations = ObservationBuilder(
            safety=config.safety,
            observation_min_tasks=config.observation_min_tasks,
            observation_max_tasks=config.observation_max_tasks,
            latency_slo_seconds=config.latency_slo_seconds,
        )

    def run(self) -> None:
        iteration = 0
        while self.config.max_iterations is None or iteration < self.config.max_iterations:
            started = time.monotonic()
            self.step(iteration=iteration)
            iteration += 1
            sleep_for = self.config.interval_seconds - (time.monotonic() - started)
            if sleep_for > 0 and (
                self.config.max_iterations is None or iteration < self.config.max_iterations
            ):
                time.sleep(sleep_for)

    def step(self, *, iteration: int) -> None:
        now = datetime.now(UTC)
        service = self.describe_service()
        desired_count = int(service.get("desiredCount", self.config.safety.min_tasks))
        running_count = int(service.get("runningCount", 0))
        cpu = self.latest_ecs_cpu_utilization()
        latency = self.latest_target_response_time()
        raw_obs = self.observations.build(
            cpu_utilization_percent=cpu,
            target_response_time_seconds=latency,
            desired_count=desired_count,
            now=now,
        )
        normalized_obs = self.vecnormalize.normalize_obs(raw_obs.reshape(1, -1))
        masks = action_mask(
            desired_count,
            self.config.safety.min_tasks,
            self.config.safety.max_tasks,
        ).reshape(1, -1)
        action, _ = self.model.predict(
            normalized_obs,
            deterministic=self.config.deterministic,
            action_masks=masks,
        )
        action_index = int(np.asarray(action).reshape(-1)[0])
        proposed_delta = int(ACTION_DELTAS[action_index])
        decision = self.safety.apply(
            current_count=desired_count,
            proposed_delta=proposed_delta,
            now=now,
        )

        if decision.should_update:
            self.ecs.update_service(
                cluster=self.config.cluster_name,
                service=self.config.service_name,
                desiredCount=decision.desired_count,
            )
        self.observations.record_scale_action(decision.desired_count - desired_count)

        self.log(
            {
                "iteration": iteration,
                "timestamp": now.isoformat(),
                "cluster": self.config.cluster_name,
                "service": self.config.service_name,
                "desired_count": desired_count,
                "running_count": running_count,
                "cpu_utilization_percent": round(cpu, 3),
                "target_response_time_seconds": round(latency, 4),
                "observation": np.round(raw_obs, 4).tolist(),
                "action_index": action_index,
                "policy_delta": proposed_delta,
                "bounded_delta": decision.bounded_delta,
                "new_desired_count": decision.desired_count,
                "updated_ecs": decision.should_update,
                "safety_intervention": decision.intervention,
            }
        )

    def describe_service(self) -> dict[str, Any]:
        response = self.ecs.describe_services(
            cluster=self.config.cluster_name,
            services=[self.config.service_name],
        )
        failures = response.get("failures", [])
        if failures:
            raise RuntimeError(f"ECS describe_services failed: {failures}")
        services = response.get("services", [])
        if not services:
            raise RuntimeError("ECS service not found")
        return services[0]

    def latest_ecs_cpu_utilization(self) -> float:
        return self.latest_metric(
            namespace="AWS/ECS",
            metric_name="CPUUtilization",
            dimensions=[
                {"Name": "ClusterName", "Value": self.config.cluster_name},
                {"Name": "ServiceName", "Value": self.config.service_name},
            ],
            statistic="Average",
            default=0.0,
        )

    def latest_target_response_time(self) -> float:
        dimensions = [{"Name": "LoadBalancer", "Value": self.config.alb_arn_suffix}]
        if self.config.target_group_arn_suffix:
            dimensions.append({"Name": "TargetGroup", "Value": self.config.target_group_arn_suffix})
        value = self.latest_metric(
            namespace="AWS/ApplicationELB",
            metric_name="TargetResponseTime",
            dimensions=dimensions,
            statistic="Average",
            default=float("nan"),
        )
        if not math.isnan(value):
            return value
        return self.latest_metric(
            namespace="AWS/ApplicationELB",
            metric_name="TargetResponseTime",
            dimensions=[{"Name": "LoadBalancer", "Value": self.config.alb_arn_suffix}],
            statistic="Average",
            default=0.0,
        )

    def latest_metric(
        self,
        *,
        namespace: str,
        metric_name: str,
        dimensions: list[dict[str, str]],
        statistic: str,
        default: float,
    ) -> float:
        end = datetime.now(UTC)
        start = end - timedelta(minutes=self.config.metric_lookback_minutes)
        response = self.cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start,
            EndTime=end,
            Period=self.config.metric_period_seconds,
            Statistics=[statistic],
        )
        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return default
        latest = max(datapoints, key=lambda point: point["Timestamp"])
        return float(latest.get(statistic, default))

    @staticmethod
    def log(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aws/demo.yaml")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_controller_config(Path(args.config))
    if args.once:
        config = ControllerConfig(
            **{
                **config.__dict__,
                "max_iterations": 1,
            }
        )
    DriftScaleAwsController(config).run()


def load_controller_config(config_path: Path) -> ControllerConfig:
    raw = load_yaml(config_path)
    aws_cfg = raw.get("aws", {})
    terraform_dir = Path(aws_cfg.get("terraform_dir", "infra/terraform"))
    region = str(aws_cfg.get("region") or terraform_output(terraform_dir, "aws_region"))
    ecs_cfg = raw.get("ecs", {})
    alb_cfg = raw.get("alb", {})
    policy_cfg = raw.get("policy", {})
    safety_cfg = raw.get("safety", {})
    controller_cfg = raw.get("controller", {})
    default_model_path = "results/sensitivity/linear/replay_stage_125/model.zip"
    default_vecnormalize_path = "results/sensitivity/linear/replay_stage_125/vecnormalize.pkl"

    return ControllerConfig(
        region=region,
        cluster_name=str(
            ecs_cfg.get("cluster_name") or terraform_output(terraform_dir, "cluster_name")
        ),
        service_name=str(
            ecs_cfg.get("service_name") or terraform_output(terraform_dir, "service_name")
        ),
        alb_arn_suffix=str(
            alb_cfg.get("arn_suffix") or terraform_output(terraform_dir, "alb_arn_suffix")
        ),
        target_group_arn_suffix=optional_value(
            alb_cfg.get("target_group_arn_suffix")
            or terraform_output(terraform_dir, "target_group_arn_suffix")
        ),
        model_path=Path(policy_cfg.get("model_path", default_model_path)),
        vecnormalize_path=Path(
            policy_cfg.get(
                "vecnormalize_path",
                default_vecnormalize_path,
            )
        ),
        deterministic=bool(policy_cfg.get("deterministic", True)),
        observation_min_tasks=int(policy_cfg.get("observation_min_tasks", 1)),
        observation_max_tasks=int(policy_cfg.get("observation_max_tasks", 20)),
        safety=SafetyConfig(
            min_tasks=int(safety_cfg.get("min_tasks", 1)),
            max_tasks=int(safety_cfg.get("max_tasks", 6)),
            cooldown_seconds=int(safety_cfg.get("cooldown_seconds", 60)),
            max_scale_delta=int(safety_cfg.get("max_scale_delta", 1)),
        ),
        interval_seconds=int(controller_cfg.get("interval_seconds", 30)),
        max_iterations=optional_int(controller_cfg.get("max_iterations", 12)),
        metric_period_seconds=int(controller_cfg.get("metric_period_seconds", 60)),
        metric_lookback_minutes=int(controller_cfg.get("metric_lookback_minutes", 10)),
        latency_slo_seconds=float(controller_cfg.get("latency_slo_seconds", 0.75)),
    )


def terraform_output(terraform_dir: Path, name: str) -> str:
    result = subprocess.run(
        ["terraform", f"-chdir={terraform_dir}", "output", "-raw", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def optional_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalize_task_count(task_count: int, min_tasks: int, max_tasks: int) -> float:
    if max_tasks == min_tasks:
        return 0.0
    return float((task_count - min_tasks) / (max_tasks - min_tasks))


if __name__ == "__main__":
    main()
