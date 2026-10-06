#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/venv-thor"
ENV_FILE="${REACHY_ENV_FILE:-$HOME/.config/reachy-assistant/env}"

umask 077

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV; create the Thor virtual environment first." >&2
  exit 1
fi

export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export REACHY_PLATFORM=thor
export REACHY_PRODUCTION_MODE="${REACHY_PRODUCTION_MODE:-false}"
token="${REACHY_WEB_API_TOKEN:-}"

if [[ "$REACHY_PRODUCTION_MODE" == "true" && ${#token} -lt 32 ]]; then
  echo "REACHY_WEB_API_TOKEN must be set to at least 32 characters in $ENV_FILE." >&2
  exit 1
fi

cd "$REPO_DIR"
"$VENV/bin/python" scripts/preflight_thor.py
exec "$VENV/bin/python" -u run_web_vision_chat.py "$@"
