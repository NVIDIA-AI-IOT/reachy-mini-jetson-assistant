#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Serve the Reachy Mini VLM on Jetson AGX Thor through vLLM.

set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
MODEL_REVISION="${MODEL_REVISION:-fee6332c1abaafb77f6f9624236c63aa2f1d0187}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
NAME="${NAME:-reachy-vllm}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm@sha256:570f9a5ffa89a772226abcc98c2d358a56ec3f755c97bc079c7f2396ffe62260}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
PULL_POLICY="${PULL_POLICY:-missing}"

mkdir -p "$CACHE_DIR"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container '$NAME' already exists. Stop it before starting another server." >&2
    exit 1
fi

docker_args=(
    --rm
    --pull "$PULL_POLICY"
    --runtime=nvidia
    --network host
    --ipc host
    --init
    --stop-timeout 30
    --security-opt no-new-privileges:true
    --pids-limit 4096
    --health-cmd "curl -fsS http://127.0.0.1:$PORT/health || exit 1"
    --health-interval 15s
    --health-timeout 5s
    --health-retries 12
    --health-start-period 180s
)
if [[ "${DETACH:-0}" == "1" ]]; then
    docker_args+=(-d)
elif [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

docker run "${docker_args[@]}" \
    --name "$NAME" \
    -v "$CACHE_DIR:/root/.cache/huggingface" \
    "$IMAGE" \
    vllm serve "$MODEL" \
        --revision "$MODEL_REVISION" \
        --served-model-name "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enable-auto-tool-choice \
        --reasoning-parser gemma4 \
        --tool-call-parser gemma4
