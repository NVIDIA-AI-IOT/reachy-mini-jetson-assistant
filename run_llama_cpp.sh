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
#   ./run_llama_cpp.sh unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
#   ./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0
#   ./run_llama_cpp.sh ./models/gemma-4-E2B-it-Q4_K_M.gguf
#
# Options (env vars):
#   PORT=8090 ./run_llama_cpp.sh ...          # custom port (default: 8080)
#   CTX=2048 ./run_llama_cpp.sh ...           # context (VLM default: 1024; text: 4096)
#   BATCH=256 ./run_llama_cpp.sh ...           # logical batch (VLM default: 128; text: 2048)
#   UBATCH=256 ./run_llama_cpp.sh ...          # physical batch (VLM default: 128; text: 512)
#   RESTART=no ./run_llama_cpp.sh ...          # Docker restart policy (default: unless-stopped)
#   REASONING=auto ./run_llama_cpp.sh ...       # enable model reasoning (default: off for spoken replies)
#   CACHE_RAM=256 ./run_llama_cpp.sh ...        # prompt-cache RAM in MiB (default: 0 on 8 GB Jetsons)
#   NP=1 ./run_llama_cpp.sh ...                # parallel slots (default: 1, use 1 for VLM)
#   NAME=my-llm ./run_llama_cpp.sh ...          # custom container name
#   EMBED=1 ./run_llama_cpp.sh ...              # run as embedding server
#   MMPROJ=./models/mmproj-F16.gguf ...         # explicit projector (must share model directory)
#   LLAMA_CPP_IMAGE=... ./run_llama_cpp.sh ... # override the JetPack-compatible image
#
# Stop:
#   docker stop assistant-llm

set -euo pipefail

MODEL="${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
PORT="${PORT:-8080}"
CTX="${CTX:-}"
BATCH="${BATCH:-}"
UBATCH="${UBATCH:-}"
RESTART="${RESTART:-unless-stopped}"
REASONING="${REASONING:-off}"
CACHE_RAM="${CACHE_RAM:-0}"
NP="${NP:-1}"
# The old r36.4/cu126 pin is a JetPack 6 image. NVIDIA AI-IOT publishes this
# Jetson Orin tag for current JetPack releases; override it to pin a digest.
IMAGE="${LLAMA_CPP_IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}"

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
    local repo="${spec%%:*}"
    local repo_name="${repo##*/}"
    local base="${repo_name%-GGUF}"
    local base_lower
    local quant_lower
    base_lower="$(echo "$base" | tr '[:upper:]' '[:lower:]')"
    quant_lower="$(echo "$quant" | tr '[:upper:]' '[:lower:]')"

    # Exact match
    if [ -f "$MODELS_DIR/$expected" ]; then
        echo "$MODELS_DIR/$expected"
        return 0
    fi

    # Some repositories vary filename case or add a suffix. Require both the
    # repository's model family and quantization so an unrelated Q4 model can
    # never be selected merely because it is already cached.
    for f in "$MODELS_DIR"/*.gguf; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in
            mmproj*) continue ;;
        esac
        local filename_lower
        filename_lower="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
        case "$filename_lower" in
            *"$base_lower"*"$quant_lower"*)
                echo "$f"
                return 0
                ;;
        esac
    done

    return 1
}

download_hf_projector() {
    local repo="$1"
    local repo_name="${repo##*/}"
    local base="${repo_name%-GGUF}"
    local remote="mmproj-F16.gguf"
    local dest="$MODELS_DIR/mmproj-${base}-F16.gguf"
    local url="https://huggingface.co/${repo}/resolve/main/${remote}"

    if [ -f "$dest" ]; then
        return 0
    fi

    # VLM repositories commonly publish this companion file. Text-only
    # repositories return a non-success status and simply skip this step.
    if wget -q --spider "$url"; then
        echo "Downloading $remote for vision..."
        if wget -c --progress=bar:force -O "$dest" "$url" 2>&1; then
            echo "✓ Downloaded $(basename "$dest")"
            return 0
        fi
        rm -f "$dest"
        echo "✗ Multimodal projector download failed."
        return 1
    fi
    return 0
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

HF_REPO=""

if [ -f "$MODEL" ]; then
    # Explicit local path
    LOCAL_MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

elif echo "$MODEL" | grep -q '/'; then
    # HuggingFace spec (user/repo:quant)
    HF_REPO="${MODEL%%:*}"
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

# Probe/download the named projector even when the model was already cached.
# Text-only repositories simply return success without creating one.
if [ -n "$HF_REPO" ]; then
    download_hf_projector "$HF_REPO"
fi

MODEL_DIR="$(dirname "$LOCAL_MODEL")"
MODEL_BASE="$(basename "$LOCAL_MODEL")"

# Use an explicit projector or auto-detect a model-family-named projector.
# Only attach an mmproj whose filename contains part of the model name,
# so e.g. mmproj-gemma-4-E2B-it-F16.gguf matches
# gemma-4-E2B-it-Q4_K_M.gguf but not an unrelated model.
MMPROJ_ARGS=""
if [ "${EMBED:-0}" != "1" ]; then
    if [ -n "${MMPROJ:-}" ]; then
        if [ ! -f "$MMPROJ" ]; then
            echo "ERROR: MMPROJ does not exist: $MMPROJ"
            exit 1
        fi
        MMPROJ_PATH="$(cd "$(dirname "$MMPROJ")" && pwd)/$(basename "$MMPROJ")"
        MMPROJ_DIR="$(dirname "$MMPROJ_PATH")"
        if [ "$MMPROJ_DIR" != "$MODEL_DIR" ]; then
            echo "ERROR: MMPROJ must be in the same directory as the model: $MODEL_DIR"
            exit 1
        fi
        MMPROJ_BASE="$(basename "$MMPROJ_PATH")"
        MMPROJ_ARGS="--mmproj /models/$MMPROJ_BASE"
        echo "Vision: $MMPROJ_BASE (explicit multimodal projector)"
    else
        # Extract model family from filename (strip quant suffix like -Q4_K_M).
        MODEL_FAMILY="$(echo "$MODEL_BASE" | sed -E 's/-[QFBqfb][0-9_A-Za-z]+\.gguf$//')"

        # Try an mmproj matching the model family; never guess a generic one.
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
    fi
fi

# Full GPU STT + VLM + TTS is memory-constrained on Orin Nano 8GB. Keep
# multimodal image batches small and valid (BATCH == UBATCH) so Gemma 4's
# 264 image tokens are decoded in several bounded chunks. Preserve llama.cpp's
# former text/embedding-sized defaults when no multimodal projector is active.
if [ -n "$MMPROJ_ARGS" ]; then
    CTX="${CTX:-1024}"
    BATCH="${BATCH:-128}"
    UBATCH="${UBATCH:-128}"
else
    CTX="${CTX:-4096}"
    BATCH="${BATCH:-2048}"
    UBATCH="${UBATCH:-512}"
fi

echo "Model : $MODEL_BASE (local)"
echo "Port  : $PORT"
echo ""
echo "Context: $CTX tokens"
echo "Batch : $BATCH / $UBATCH logical / physical"
echo "Restart: $RESTART"
echo "Reasoning: $REASONING"
echo "Prompt cache RAM: ${CACHE_RAM} MiB"

docker run -d \
    --name "$NAME" \
    --runtime=nvidia \
    --restart "$RESTART" \
    -p "${PORT}:8080" \
    -v "$MODEL_DIR:/models:ro" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "$IMAGE" \
    llama-server \
    -m "/models/$MODEL_BASE" \
    $MMPROJ_ARGS \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 -c "$CTX" -b "$BATCH" -ub "$UBATCH" -np "$NP" -fa on \
    --cache-reuse 256 --cache-ram "$CACHE_RAM" --reasoning "$REASONING" $EXTRA_ARGS

echo "✓ Container '$NAME' started."
echo ""
echo "  API  : http://localhost:${PORT}/v1/chat/completions"
echo "  Logs : docker logs -f $NAME"
echo "  Stop : docker stop $NAME"
