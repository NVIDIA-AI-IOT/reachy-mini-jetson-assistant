#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${VENV:-$repo/.venv}"
if [[ -z "${VENV:-}" && ! -x "$venv/bin/python" && -x "$repo/venv/bin/python" ]]; then
  venv="$repo/venv"
fi

[[ -x "$venv/bin/python" ]] || {
  echo "Missing $venv; follow docs/JETPACK_7_2_SETUP.md (or SETUP.md for JetPack 6)." >&2
  exit 1
}

export REACHY_PLATFORM=orin_nano
export REACHY_PRODUCTION_MODE="${REACHY_PRODUCTION_MODE:-false}"
export PATH="$venv/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$repo"
exec "$venv/bin/python" -u run_web_vision_chat.py "$@"
