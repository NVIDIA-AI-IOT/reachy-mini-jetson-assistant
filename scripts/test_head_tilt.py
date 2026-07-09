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

"""Head-tilt diagnostic for Reachy Mini.

Streams a sequence of distinct head orientations (yaw, pitch, roll)
using exactly the same API the face tracker uses (create_head_pose +
set_target). Watch the robot and confirm each labeled angle produces
the expected motion. This isolates the hardware/SDK from the tracking
control loop.

Usage:
  source venv/bin/activate
  python3 scripts/test_head_tilt.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from app.config import Config
from app.reachy import connect as connect_reachy
from reachy_mini.utils import create_head_pose

console = Console()
_NEUTRAL_ANT = np.array([0.0, 0.0], dtype=np.float64)


def stream_pose(reachy, label, *, yaw=0.0, pitch=0.0, roll=0.0, secs=2.5, hz=100):
    """Hold a head pose for `secs` by streaming the target at `hz`."""
    console.print(f"  -> {label}  (yaw={yaw:+.0f}  pitch={pitch:+.0f}  roll={roll:+.0f})")
    pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
    period = 1.0 / hz
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        try:
            reachy.set_target(head=pose, antennas=_NEUTRAL_ANT, body_yaw=0.0)
        except Exception as e:
            console.print(f"    [red]set_target failed: {e}[/red]")
            return
        time.sleep(period)


def main():
    config = Config.load()
    reachy = connect_reachy(config, console)
    if reachy is None:
        console.print("[red]No robot connection — is the app still holding it?[/red]")
        return

    sequence = [
        ("CENTER (neutral)",            dict(secs=1.5)),
        ("YAW LEFT  (yaw = +25)",       dict(yaw=+25)),
        ("CENTER",                      dict(secs=1.0)),
        ("YAW RIGHT (yaw = -25)",       dict(yaw=-25)),
        ("CENTER",                      dict(secs=1.0)),
        ("PITCH UP   (pitch = +18)",    dict(pitch=+18)),
        ("CENTER",                      dict(secs=1.0)),
        ("PITCH DOWN (pitch = -18)",    dict(pitch=-18)),
        ("CENTER",                      dict(secs=1.0)),
        ("ROLL  (roll = +20)",          dict(roll=+20)),
        ("CENTER",                      dict(secs=1.0)),
        ("DIAGONAL (yaw +20, pitch +12)", dict(yaw=+20, pitch=+12)),
        ("CENTER (return)",             dict(secs=1.5)),
    ]

    console.print("\n[bold cyan]Head tilt diagnostic[/bold cyan] — watch the robot for each step.\n")
    try:
        for label, kwargs in sequence:
            stream_pose(reachy, label, **kwargs)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        stream_pose(reachy, "settle neutral", secs=1.0)
        console.print("\n[green]Done.[/green] Note which physical direction each label produced.")


if __name__ == "__main__":
    main()
