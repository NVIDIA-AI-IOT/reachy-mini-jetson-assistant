#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect to the physical Reachy Mini through the app's production path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from app.config import Config
from app.reachy import connect


def main() -> int:
    robot = connect(Config.load(), Console())
    if robot is None:
        print("REACHY_CONNECTION_FAILED")
        return 1
    print("REACHY_CONNECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
