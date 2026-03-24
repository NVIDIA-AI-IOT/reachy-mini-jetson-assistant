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

"""Continuous face tracking for Reachy Mini.

Adapted from Pollen Robotics' reachy_mini_conversation_app/camera_worker.py.
Runs YuNet face detection at ~30 Hz and feeds proportional yaw/pitch
offsets into MovementManager as a secondary additive layer.
"""

import math
import threading
import time
from typing import Optional

import numpy as np

from reachy_mini.utils import create_head_pose

from app.camera import Camera
from app.emotion import EmotionDetector
from app.movement_manager import MovementManager

MAX_YAW_DEG = 25.0
MAX_PITCH_DEG = 15.0
GAIN = 0.6

_FACE_LOST_DELAY = 2.0
_FACE_LOST_BLEND_SECS = 1.0

_IDENTITY_4x4 = np.eye(4, dtype=np.float64)


class FaceTracker:
    """Background thread that continuously tracks the largest face.

    Reads raw BGR frames from the camera, runs YuNet detection via
    EmotionDetector.detect_face(), and converts the face centre to
    proportional yaw/pitch offsets fed into the MovementManager's
    secondary offset layer.

    Visual servoing converges naturally because the camera rides on
    the head — no camera intrinsics needed.

    Face-lost behaviour (matches Pollen): 2 s delay, then 1 s linear
    interpolation back to neutral offset.
    """

    def __init__(
        self,
        camera: Camera,
        detector: EmotionDetector,
        manager: MovementManager,
        fps: float = 30.0,
    ):
        self._camera = camera
        self._detector = detector
        self._manager = manager
        self._period = 1.0 / fps

        self._enabled = True
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._last_offset = _IDENTITY_4x4.copy()
        self._face_lost_time: Optional[float] = None
        self._tracking_active = False

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="face-tracker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._manager.clear_face_offsets()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._manager.clear_face_offsets()
            self._tracking_active = False

    @property
    def is_tracking(self) -> bool:
        return self._tracking_active

    # ── main loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            t_start = time.monotonic()

            if self._enabled:
                self._tick()

            elapsed = time.monotonic() - t_start
            sleep_time = self._period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self) -> None:
        frame = self._camera.read_raw_live()
        if frame is None:
            return

        box = self._detector.detect_face(frame)
        now = time.monotonic()

        if box is not None:
            self._face_lost_time = None
            self._tracking_active = True

            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            face_cx = (x1 + x2) / 2.0
            face_cy = (y1 + y2) / 2.0

            offset_x = (face_cx - w / 2.0) / (w / 2.0)
            offset_y = (face_cy - h / 2.0) / (h / 2.0)

            yaw_deg = -offset_x * MAX_YAW_DEG * GAIN
            pitch_deg = -offset_y * MAX_PITCH_DEG * GAIN

            offset_mat = create_head_pose(
                yaw=yaw_deg, pitch=pitch_deg, degrees=True,
            )
            self._last_offset = offset_mat
            self._manager.set_face_offsets(offset_mat)
        else:
            self._handle_face_lost(now)

    def _handle_face_lost(self, now: float) -> None:
        if self._face_lost_time is None:
            self._face_lost_time = now
            return

        elapsed = now - self._face_lost_time

        if elapsed < _FACE_LOST_DELAY:
            return

        blend_t = min(
            (elapsed - _FACE_LOST_DELAY) / _FACE_LOST_BLEND_SECS, 1.0,
        )

        blended = _blend_offset_to_identity(self._last_offset, blend_t)
        self._manager.set_face_offsets(blended)

        if blend_t >= 1.0:
            self._last_offset = _IDENTITY_4x4.copy()
            self._tracking_active = False


def _blend_offset_to_identity(
    offset: np.ndarray, t: float,
) -> np.ndarray:
    """Linearly blend a 4x4 offset matrix toward identity.

    Translation is lerped, rotation is interpolated via rotvec scaling.
    """
    from scipy.spatial.transform import Rotation as R

    result = np.eye(4, dtype=np.float64)

    result[:3, 3] = offset[:3, 3] * (1.0 - t)

    rot = R.from_matrix(offset[:3, :3])
    rotvec = rot.as_rotvec()
    blended_rot = R.from_rotvec(rotvec * (1.0 - t))
    result[:3, :3] = blended_rot.as_matrix()

    return result
