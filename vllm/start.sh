#!/bin/sh
# vLLM launcher. The admin "Deploy model" button writes /config/llm/llm.env and
# restarts this container; without that file the compose defaults are used.
set -eu

CFG=/config/llm/llm.env
if [ -f "$CFG" ]; then
    . "$CFG"
    EXTRA="${LLM_EXTRA_ARGS:-}"
else
    EXTRA="${DEFAULT_LLM_EXTRA_ARGS:-}"
fi
MODEL="${LLM_MODEL:-$DEFAULT_LLM_MODEL}"
MAXLEN="${LLM_MAX_MODEL_LEN:-$DEFAULT_LLM_MAX_MODEL_LEN}"
UTIL="${LLM_GPU_UTIL:-$DEFAULT_LLM_GPU_UTIL}"

echo "[start.sh] model=$MODEL max_model_len=$MAXLEN gpu_util=$UTIL extra=$EXTRA"
# shellcheck disable=SC2086  # EXTRA is intentionally word-split
exec vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --served-model-name chat \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --enable-sleep-mode \
    --enable-prefix-caching \
    --disable-uvicorn-access-log \
    $EXTRA
