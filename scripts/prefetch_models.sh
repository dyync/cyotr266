#!/usr/bin/env bash
# Pre-download the default models into the shared HF cache (${MODELS_DIR:-/models}/hf)
# so the first user request does not wait for ~45 GB of downloads.
# Runs inside the media image (uid 1000, correct HF_HOME).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

models=(
  "${LLM_MODEL:-RedHatAI/Qwen3.5-9B-FP8-dynamic}"
  "${IMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
  "${VIDEO_MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
  "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
)
docker compose up --no-deps init >/dev/null
for m in "${models[@]}"; do
  echo "==> $m"
  docker compose run --rm --no-deps --entrypoint "" video hf download "$m" >/dev/null
done
echo "done. Cache size:"; du -sh "${MODELS_DIR:-/models}/hf"
