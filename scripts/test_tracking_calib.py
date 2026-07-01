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

"""Face-tracking sign calibration for Reachy Mini.

Keep your face centered in front of the robot. The script nudges the
head by a known yaw/pitch and measures how the detected face shifts in
the camera image. From the measured d(pixel)/d(angle) it prints the
correct control signs for the tracker.

Usage:
  source venv/bin/activate
  python3 scripts/test_tracking_calib.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector
from app.reachy import connect as connect_reachy, kill_stale_camera_holders
from reachy_mini.utils import create_head_pose

console = Console()
_NEUTRAL_ANT = np.array([0.0, 0.0], dtype=np.float64)


def hold(reachy, yaw=0.0, pitch=0.0, secs=1.5, hz=100):
    pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        reachy.set_target(head=pose, antennas=_NEUTRAL_ANT, body_yaw=0.0)
        time.sleep(1.0 / hz)


def measure_face(cam, detector, samples=12):
    """Return (cx, cy, w, h) averaged over detected frames, or None."""
    xs, ys, ws, hs = [], [], [], []
    for _ in range(samples):
        frame = cam.read_raw_live()
        if frame is not None:
            box = detector.detect(frame)
            if box is not None:
                x1, y1, x2, y2 = box
                xs.append((x1 + x2) / 2.0)
                ys.append((y1 + y2) / 2.0)
                hh, ww = frame.shape[:2]
                ws.append(ww); hs.append(hh)
        time.sleep(0.05)
    if not xs:
        return None
    return (np.median(xs), np.median(ys), np.median(ws), np.median(hs))


def main():
    config = Config.load()
    reachy = connect_reachy(config, console)
    if reachy is None:
        console.print("[red]No robot connection.[/red]")
        return

    kill_stale_camera_holders(config.vision.camera_device, console)
    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width, height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality, capture_fps=config.vision.capture_fps,
    )
    if not cam.start():
        console.print("[red]Camera failed to start.[/red]")
        return
    detector = FaceDetector()
    if not detector.load():
        console.print("[red]Face detector failed to load.[/red]")
        cam.close(); return

    console.print("\n[bold cyan]Tracking sign calibration[/bold cyan]")
    console.print("Keep your face centered in front of the robot. Starting in 3s...\n")
    hold(reachy, secs=3.0)

    base = measure_face(cam, detector)
    if base is None:
        console.print("[red]No face detected. Sit in front of the camera and rerun.[/red]")
        cam.close(); return
    console.print(f"  baseline face center: cx={base[0]:.0f}, cy={base[1]:.0f}  (frame {base[2]:.0f}x{base[3]:.0f})")

    # ── Yaw probe ────────────────────────────────────────────────
    yaw_probe = 12.0
    hold(reachy, yaw=yaw_probe, secs=1.5)
    yaw_meas = measure_face(cam, detector)
    hold(reachy, secs=1.2)  # return to neutral

    # ── Pitch probe ──────────────────────────────────────────────
    pitch_probe = 12.0
    hold(reachy, pitch=pitch_probe, secs=1.5)
    pitch_meas = measure_face(cam, detector)
    hold(reachy, secs=1.5)  # return to neutral

    cam.close()

    console.print("\n[bold]Results[/bold]")
    if yaw_meas:
        dcx = yaw_meas[0] - base[0]
        console.print(f"  yaw +{yaw_probe:.0f}\u00b0 (head LEFT)  -> face cx moved {dcx:+.0f}px "
                      f"({'right' if dcx > 0 else 'left'} in image)")
        # To center: drive err_x->0. err_x = (cx-w/2)/(w/2).
        # delta_yaw = SIGN * err_x must reduce cx-error.
        # d(cx)/d(yaw) = dcx/yaw_probe ; to reduce a positive err_x we need
        # delta_yaw with sign = -sign(d(cx)/d(yaw)).
        yaw_sign = -1.0 if dcx > 0 else 1.0
        console.print(f"  => correct yaw control: delta_yaw = {yaw_sign:+.0f} * gain * err_x")
    else:
        console.print("  [yellow]yaw probe: face not detected after move[/yellow]")

    if pitch_meas:
        dcy = pitch_meas[1] - base[1]
        console.print(f"  pitch +{pitch_probe:.0f}\u00b0 (head UP)    -> face cy moved {dcy:+.0f}px "
                      f"({'down' if dcy > 0 else 'up'} in image)")
        pitch_sign = -1.0 if dcy > 0 else 1.0
        console.print(f"  => correct pitch control: delta_pitch = {pitch_sign:+.0f} * gain * err_y")
    else:
        console.print("  [yellow]pitch probe: face not detected after move[/yellow]")

    console.print("\n[green]Done.[/green]")


if __name__ == "__main__":
    main()
