#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$HOME/.config/reachy-assistant"

install -d -m 700 "$env_dir"
if [[ ! -f "$env_dir/env" ]]; then
  install -m 600 "$repo/deploy/reachy.env.example" "$env_dir/env"
  echo "Created $env_dir/env from the Thor defaults. Review it before enabling services."
fi

sudo install -m 644 "$repo/deploy/systemd/reachy-vllm.service" /etc/systemd/system/
sudo install -m 644 "$repo/deploy/systemd/reachy-assistant.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo "Units installed but not enabled or started."
echo "Validate first, then run: sudo systemctl enable --now reachy-assistant.service"
