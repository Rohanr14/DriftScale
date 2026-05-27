#!/usr/bin/env bash
set -euo pipefail

TF_DIR="${TF_DIR:-infra/terraform}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_IMAGE="${LOCAL_IMAGE:-driftscale-demo-app:${IMAGE_TAG}}"

REGION="$(terraform -chdir="${TF_DIR}" output -raw aws_region)"
REPOSITORY_URL="$(terraform -chdir="${TF_DIR}" output -raw ecr_repository_url)"
REGISTRY="${REPOSITORY_URL%/*}"

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

docker build -t "${LOCAL_IMAGE}" -f app/Dockerfile app
docker tag "${LOCAL_IMAGE}" "${REPOSITORY_URL}:${IMAGE_TAG}"
docker push "${REPOSITORY_URL}:${IMAGE_TAG}"

echo "Pushed ${REPOSITORY_URL}:${IMAGE_TAG}"
