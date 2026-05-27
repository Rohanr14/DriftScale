"""Run k6 load generation and the AWS controller together for the demo."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from driftscale.aws.controller import terraform_output
from driftscale.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aws/demo.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    terraform_dir = Path(config.get("aws", {}).get("terraform_dir", "infra/terraform"))
    alb_url = terraform_output(terraform_dir, "alb_url")
    k6_script = Path(config.get("load", {}).get("k6_script", "load/k6_script.js"))
    if not k6_script.exists():
        raise FileNotFoundError(k6_script)

    env = os.environ.copy()
    env["BASE_URL"] = alb_url
    k6_process = subprocess.Popen(["k6", "run", str(k6_script)], env=env)
    controller_process = subprocess.Popen(
        [sys.executable, "-m", "driftscale.aws.controller", "--config", str(config_path)]
    )

    def stop_children(*_args) -> None:
        for process in (controller_process, k6_process):
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)

    try:
        controller_code = controller_process.wait()
        if k6_process.poll() is None:
            k6_process.terminate()
        k6_process.wait(timeout=15)
        raise SystemExit(controller_code)
    finally:
        stop_children()


if __name__ == "__main__":
    main()
