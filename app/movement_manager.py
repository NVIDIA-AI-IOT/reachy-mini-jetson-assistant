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

"""100 Hz motion controller for Reachy Mini.

Owns all robot motion through a single set_target() call per tick. The
face tracker pushes target angles:
    body_yaw   — rotate the whole base to face/search (large range)
    head pitch — tilt the head up/down for vertical centering
    head yaw   — yaw trim relative to the body/camera scan direction

Important SDK detail: set_target_head_pose() solves a world-space head
pose while also receiving body_yaw for IK. If we command body_yaw but keep
head pose yaw fixed at 0, the neck can counter-rotate and the camera keeps
looking at nearly the same field of view. To make body yaw actually scan
the camera, the commanded head pose yaw must include body_yaw too.

This loop eases the current pose toward those targets so intermittent
detections turn into smooth motion. Antennas stay at rest.
"""

import threading
import time
from typing import Optional

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import delta_angle_between_mat_rot, linear_pose_interpolation

_TICK_HZ = 100
_TICK_PERIOD = 1.0 / _TICK_HZ

# Per-tick fraction of the remaining error to close (exponential ease).
# 100 Hz loop. Higher smoothing makes face tracking feel responsive while
# the detector deadzone prevents constant micro-corrections around center.
_SMOOTHING = 0.38
_SEND_EPS_DEG = 0.25
_SETTLED_EPS_DEG = 0.40
_ERROR_LOG_PERIOD_SECS = 2.0

_NEUTRAL_ANTENNAS = np.array([0.0, 0.0], dtype=np.float64)


class MovementManager:
    """100 Hz controller easing body_yaw / head pitch / head yaw to targets.

    Public API (all thread-safe):
        set_targets(body_yaw_deg, head_pitch_deg, head_yaw_deg)
        reset()           — ease everything back to neutral
        start() / stop()  — lifecycle
    """

    def __init__(
        self,
        reachy,
        *,
        pose_smoothing: float = 0.18,
        pose_max_step_deg: float = 6.0,
    ):
        self._reachy = reachy
        self._pose_smoothing = max(0.01, min(1.0, pose_smoothing))
        self._pose_max_step_rad = np.deg2rad(max(0.5, pose_max_step_deg))
        self._lock = threading.Lock()
        # targets
        self._t_body = 0.0
        self._t_pitch = 0.0
        self._t_yaw = 0.0
        self._t_head_pose: Optional[np.ndarray] = None
        self._hold_motion = False
        self._target_version = 0
        # current (smoothed)
        self._c_body = 0.0
        self._c_pitch = 0.0
        self._c_yaw = 0.0
        self._c_head_pose: Optional[np.ndarray] = None
        self._last_sent: Optional[tuple[float, float, float]] = None
        self._last_sent_head_pose: Optional[np.ndarray] = None
        self._last_sent_version = -1
        self._last_error: Optional[str] = None
        self._last_error_log_time = 0.0
        self._last_pose_log_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="motion-100hz")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── public API ────────────────────────────────────────────────

    def set_targets(self, body_yaw_deg: float, head_pitch_deg: float, head_yaw_deg: float) -> None:
        with self._lock:
            next_targets = (body_yaw_deg, head_pitch_deg, head_yaw_deg)
            if self._t_head_pose is not None or next_targets != (self._t_body, self._t_pitch, self._t_yaw):
                self._target_version += 1
            self._t_head_pose = None
            self._hold_motion = False
            self._c_head_pose = None
            self._t_body, self._t_pitch, self._t_yaw = next_targets

    def set_head_pose_target(self, head_pose: np.ndarray) -> None:
        """Set an SDK-computed 4x4 head pose target."""
        with self._lock:
            if self._t_head_pose is None or not np.allclose(head_pose, self._t_head_pose, atol=1e-4):
                self._target_version += 1
            self._t_head_pose = head_pose.copy()
            self._hold_motion = False

    def hold_current(self) -> None:
        """Stop sending new targets and let the robot hold its last pose."""
        with self._lock:
            if self._t_head_pose is not None or not self._hold_motion:
                self._target_version += 1
            self._t_head_pose = None
            self._hold_motion = True
            self._last_sent_head_pose = None

    def reset(self) -> None:
        with self._lock:
            self._t_head_pose = None
            self._hold_motion = False
        self.set_targets(0.0, 0.0, 0.0)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ── 100 Hz loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            t_start = time.monotonic()
            self._tick()
            sleep_time = _TICK_PERIOD - (time.monotonic() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self) -> None:
        with self._lock:
            t_body, t_pitch, t_yaw = self._t_body, self._t_pitch, self._t_yaw
            t_head_pose = None if self._t_head_pose is None else self._t_head_pose.copy()
            hold_motion = self._hold_motion
            target_version = self._target_version

        if hold_motion:
            return

        if t_head_pose is not None:
            if self._c_head_pose is None:
                try:
                    self._c_head_pose = self._reachy.get_current_head_pose()
                except Exception:
                    self._c_head_pose = t_head_pose.copy()

            angle_delta = delta_angle_between_mat_rot(
                self._c_head_pose[:3, :3],
                t_head_pose[:3, :3],
            )
            step_ratio = self._pose_smoothing
            if angle_delta > self._pose_max_step_rad:
                step_ratio = min(step_ratio, self._pose_max_step_rad / angle_delta)
            next_head_pose = linear_pose_interpolation(self._c_head_pose, t_head_pose, step_ratio)
            now = time.monotonic()
            if now - self._last_pose_log_time >= 1.0:
                print(
                    "MovementManager: "
                    f"pose_target_delta={np.rad2deg(angle_delta):.1f}deg "
                    f"step_ratio={step_ratio:.2f}"
                )
                self._last_pose_log_time = now

            if (
                target_version == self._last_sent_version
                and self._last_sent_head_pose is not None
                and np.allclose(next_head_pose, self._last_sent_head_pose, atol=1e-3)
            ):
                return
            try:
                self._reachy.set_target(head=next_head_pose, antennas=_NEUTRAL_ANTENNAS)
                self._c_head_pose = next_head_pose
                self._last_sent_head_pose = next_head_pose
                self._last_sent_version = target_version
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                now = time.monotonic()
                if now - self._last_error_log_time >= _ERROR_LOG_PERIOD_SECS:
                    print(f"MovementManager: set_target failed: {e}")
                    self._last_error_log_time = now
            return

        target_delta = max(abs(t_body - self._c_body), abs(t_pitch - self._c_pitch), abs(t_yaw - self._c_yaw))
        if target_delta < _SETTLED_EPS_DEG:
            self._c_body, self._c_pitch, self._c_yaw = t_body, t_pitch, t_yaw
        else:
            self._c_body += (t_body - self._c_body) * _SMOOTHING
            self._c_pitch += (t_pitch - self._c_pitch) * _SMOOTHING
            self._c_yaw += (t_yaw - self._c_yaw) * _SMOOTHING

        pose = (self._c_body, self._c_pitch, self._c_yaw)
        target_delta = max(abs(t_body - self._c_body), abs(t_pitch - self._c_pitch), abs(t_yaw - self._c_yaw))
        if self._last_sent is not None:
            sent_delta = max(abs(a - b) for a, b in zip(pose, self._last_sent))
            if (
                target_version == self._last_sent_version
                and sent_delta < _SEND_EPS_DEG
                and target_delta < _SETTLED_EPS_DEG
            ):
                return

        try:
            # Head pose yaw is world-space for the IK. Include body yaw so
            # body rotation changes camera direction instead of being
            # cancelled by the neck solver. self._c_yaw remains a trim on
            # top of the body's scan/tracking direction.
            head_world_yaw = self._c_body + self._c_yaw
            head = create_head_pose(yaw=head_world_yaw, pitch=self._c_pitch, degrees=True)
            self._reachy.set_target(
                head=head,
                antennas=_NEUTRAL_ANTENNAS,
                body_yaw=np.radians(self._c_body),
            )
            self._last_sent = pose
            self._last_sent_head_pose = None
            self._last_sent_version = target_version
            self._last_error = None
        except Exception as e:
            self._last_error = str(e)
            now = time.monotonic()
            if now - self._last_error_log_time >= _ERROR_LOG_PERIOD_SECS:
                print(f"MovementManager: set_target failed: {e}")
                self._last_error_log_time = now
