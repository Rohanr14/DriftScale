.PHONY: setup test lint fetch-data preprocess calibrate train-baseline drift-experiment sensitivity-suite post-eval deploy-demo live-demo destroy-demo plots phase1 random-run eval clean

setup:
	uv sync --extra dev

test:
	uv run --extra dev pytest tests -q

lint:
	uv run --extra dev ruff check driftscale tests scripts

fetch-data:
	uv run python scripts/fetch_azure_sample.py

preprocess:
	uv run python scripts/preprocess_traces.py --config configs/env/azure_v1.yaml

calibrate: preprocess
	uv run python scripts/calibrate_baselines.py --train-config configs/train/ppo.yaml

train-baseline: calibrate
	uv run --extra train python -m driftscale.agents.train_ppo --config configs/train/ppo.yaml
	uv run python scripts/make_plots.py

drift-experiment: calibrate
	uv run --extra train python scripts/run_drift_experiment.py

sensitivity-suite:
	@uv run --extra train python scripts/run_sensitivity_analysis.py

post-eval:
	uv run --extra train python scripts/run_sensitivity_analysis.py --plot-only
	uv run --extra train python scripts/plot_episode_rollout.py --plot-only

deploy-demo:
	cd infra/terraform && terraform init
	cd infra/terraform && terraform apply -target=aws_budgets_budget.demo -target=aws_ecr_repository.app
	./scripts/build_push_demo_image.sh
	cd infra/terraform && terraform apply

live-demo:
	uv run --extra aws --extra train python scripts/run_live_demo.py --config configs/aws/demo.yaml

destroy-demo:
	cd infra/terraform && terraform destroy

plots:
	uv run python scripts/make_plots.py

phase1:
	uv run python scripts/eval_phase1.py --config configs/env/synthetic.yaml

random-run:
	uv run python scripts/run_random_policy.py --config configs/env/synthetic.yaml

eval: phase1

clean:
	rm -rf .pytest_cache .ruff_cache results/phase1 results/caches results/calibration results/ppo_vanilla results/drift_experiment results/sensitivity media/cost_vs_slo.png media/continuous_forgetting.png media/episode_rollout.png
