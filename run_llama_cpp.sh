#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run llama.cpp server (GPU) — pass any HuggingFace model or local GGUF.
#
# Models are stored in ./models/ for fully offline operation.
# On first run with a HF spec, models are downloaded and saved locally.
# All subsequent runs load from disk — no internet required.
#
# Usage:
#   ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M
#   ./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0
#   ./run_llama_cpp.sh ./models/Cosmos-Reason2-2B-Q4_K_M.gguf
#
# Options (env vars):
#   PORT=8090 ./run_llama_cpp.sh ...          # custom port (default: 8080)
#   CTX=4096 ./run_llama_cpp.sh ...           # custom context size (default: 4096)
#   NP=1 ./run_llama_cpp.sh ...              # parallel slots (default: 1, use 1 for VLM)
#   NAME=my-llm ./run_llama_cpp.sh ...        # custom container name
#   EMBED=1 ./run_llama_cpp.sh ...            # run as embedding server
#
# Stop:
#   docker stop assistant-llm

set -e

MODEL="${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NP="${NP:-1}"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:b8095-r36.4-tegra-aarch64-cu126-22.04"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

if [ "${EMBED:-0}" = "1" ]; then
    NAME="${NAME:-assistant-embed}"
    EXTRA_ARGS="--embeddings"
else
    NAME="${NAME:-assistant-llm}"
    EXTRA_ARGS=""
fi

# Stop existing container with same name
if [ "$(docker ps -aq -f name=^${NAME}$)" ]; then
    echo "Stopping existing $NAME..."
    docker stop "$NAME" > /dev/null 2>&1 || true
    docker rm "$NAME" > /dev/null 2>&1 || true
fi

# ── HuggingFace spec helpers ────────────────────────────────────
# Parse "user/repo:quant" → derive expected local filename and download URL.

hf_expected_filename() {
    local spec="$1"
    local repo="${spec%%:*}"
    local quant="${spec##*:}"
    local repo_name="${repo##*/}"
    local base="${repo_name%-GGUF}"
    base="${base%-$quant}"
    echo "${base}-${quant}.gguf"
}

find_local_model() {
    local spec="$1"
    local expected
    expected="$(hf_expected_filename "$spec")"
    local quant="${spec##*:}"

    # Exact match
    if [ -f "$MODELS_DIR/$expected" ]; then
        echo "$MODELS_DIR/$expected"
        return 0
    fi

    # Fuzzy: any GGUF in models/ containing the quant string (skip mmproj files)
    local match
    for f in "$MODELS_DIR"/*.gguf; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in
            mmproj*) continue ;;
        esac
        case "$(basename "$f")" in
            *"$quant"*|*"$(echo "$quant" | tr '[:upper:]' '[:lower:]')"*)
                echo "$f"
                return 0
                ;;
        esac
    done

    return 1
}

download_hf_model() {
    local spec="$1"
    local repo="${spec%%:*}"
    local expected
    expected="$(hf_expected_filename "$spec")"
    local url="https://huggingface.co/${repo}/resolve/main/${expected}"
    local dest="$MODELS_DIR/$expected"

    echo "Downloading $expected ..."
    echo "  URL : $url"
    echo "  Dest: $dest"
    if wget -c --progress=bar:force -O "$dest" "$url" 2>&1; then
        echo "✓ Downloaded $expected"
        return 0
    fi
    rm -f "$dest"
    echo "✗ Download failed. Check your internet connection."
    return 1
}

# ── Resolve model to a local file ───────────────────────────────

if [ -f "$MODEL" ]; then
    # Explicit local path
    LOCAL_MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

elif echo "$MODEL" | grep -q '/'; then
    # HuggingFace spec (user/repo:quant)
    LOCAL_MODEL="$(find_local_model "$MODEL" 2>/dev/null)" || {
        echo "Model not found in $MODELS_DIR — downloading..."
        download_hf_model "$MODEL"
        LOCAL_MODEL="$(find_local_model "$MODEL")" || {
            echo "ERROR: could not resolve model after download."
            exit 1
        }
    }
    echo "Model : $(basename "$LOCAL_MODEL") (local cache)"
else
    echo "ERROR: '$MODEL' is not a local file or HuggingFace spec (user/repo:quant)."
    exit 1
fi

MODEL_DIR="$(dirname "$LOCAL_MODEL")"
MODEL_BASE="$(basename "$LOCAL_MODEL")"

# Auto-detect multimodal projector (mmproj) for VLMs.
# Only attach an mmproj whose filename contains part of the model name,
# so e.g. mmproj-Cosmos-Reason2-2B-F16.gguf matches Cosmos-Reason2-2B-Q4_K_M.gguf
# but not Qwen3.5-0.8B-Q5_K_M.gguf.
MMPROJ_ARGS=""
if [ "${EMBED:-0}" != "1" ]; then
    # Extract model family from filename (strip quant suffix like -Q4_K_M)
    MODEL_FAMILY="$(echo "$MODEL_BASE" | sed -E 's/-[QFBqfb][0-9_]+[A-Za-z]*\.gguf$//')"

    # Try mmproj matching the model family first
    for f in "$MODEL_DIR"/mmproj*.gguf; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in
            *"$MODEL_FAMILY"*)
                MMPROJ_BASE="$(basename "$f")"
                MMPROJ_ARGS="--mmproj /models/$MMPROJ_BASE"
                echo "Vision: $MMPROJ_BASE (multimodal projector)"
                break
                ;;
        esac
    done

    # Fall back to any generic mmproj (no model name in filename, e.g. mmproj-F16.gguf)
    if [ -z "$MMPROJ_ARGS" ]; then
        for f in "$MODEL_DIR"/mmproj*.gguf; do
            [ -f "$f" ] || continue
            fname="$(basename "$f")"
            # Generic mmproj: short name like mmproj-F16.gguf or mmproj-BF16.gguf
            case "$fname" in
                mmproj-[FBfb]*.gguf)
                    MMPROJ_BASE="$fname"
                    MMPROJ_ARGS="--mmproj /models/$MMPROJ_BASE"
                    echo "Vision: $MMPROJ_BASE (generic multimodal projector)"
                    break
                    ;;
            esac
        done
    fi
fi

echo "Model : $MODEL_BASE (local)"
echo "Port  : $PORT"
echo ""

docker run -d \
    --name "$NAME" \
    --runtime=nvidia \
    -p "${PORT}:8080" \
    -v "$MODEL_DIR:/models:ro" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "$IMAGE" \
    llama-server \
    -m "/models/$MODEL_BASE" \
    $MMPROJ_ARGS \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 -c "$CTX" -np "$NP" -fa on --cache-reuse 256 $EXTRA_ARGS

echo "✓ Container '$NAME' started."
echo ""
echo "  API  : http://localhost:${PORT}/v1/chat/completions"
echo "  Logs : docker logs -f $NAME"
echo "  Stop : docker stop $NAME"
