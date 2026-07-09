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

"""Camera — USB webcam with shared latest frame and VLM ring buffer.

The background thread owns hardware reads and publishes frames into a
timestamped ring buffer. Consumers such as the web UI, face tracker, and VLM
capture use the latest buffered frame instead of competing for VideoCapture.
"""

import base64
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    from reachy_mini.media.camera_utils import find_camera
    from reachy_mini.media.camera_constants import CameraResolution
    from reachy_mini.media.camera_utils import scale_intrinsics
    HAS_REACHY_CAM = True
except ImportError:
    HAS_REACHY_CAM = False

MAX_SPEECH_SECS = 16
PRE_SPEECH_SECS = 0.5


class Camera:
    """V4L2 USB webcam with a background capture thread and timestamped
    ring buffer sized to cover the maximum speech duration plus lookback."""

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 80,
        capture_fps: float = 3.0,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.capture_fps = capture_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self.camera_specs = None
        self.camera_K: Optional[np.ndarray] = None
        self.camera_D: Optional[np.ndarray] = None
        self._cap_lock = threading.Lock()
        # Reachy's camera can ignore the requested 640x480 mode and return
        # 1920x1080 BGR frames. Keeping a 16.5 second raw ring at 10 FPS would
        # retain about 979 MiB. Preserve the temporal window as JPEG bytes and
        # keep only the newest frame raw for face tracking.
        self._ring: deque[tuple[float, bytes]] = deque(
            maxlen=max(1, int(capture_fps * (MAX_SPEECH_SECS + PRE_SPEECH_SECS)))
        )
        self._latest: Optional[tuple[float, np.ndarray]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._alive = False
        self._thread: Optional[threading.Thread] = None
        self._actual_fps: float = 0.0

    def open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True

        # Try Reachy Mini SDK camera detection first (finds correct device by USB VID/PID)
        if HAS_REACHY_CAM:
            try:
                cap, specs = find_camera()
                if cap is not None and cap.isOpened():
                    self._cap = cap
                    self.camera_specs = specs
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._sync_actual_frame_geometry()
                    return True
            except Exception:
                pass

        # Fallback: open by device index
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._sync_actual_frame_geometry()
        return True

    def _sync_actual_frame_geometry(self) -> None:
        """Record the mode accepted by the camera and calibrate for that mode."""
        if self._cap is None:
            return
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0:
            self.width = actual_width
        if actual_height > 0:
            self.height = actual_height
        self._set_camera_calibration(width=self.width, height=self.height)

    def _set_camera_calibration(self, *, width: int, height: int) -> None:
        """Scale SDK camera intrinsics to the frame size used by OpenCV."""
        specs = self.camera_specs
        if specs is None:
            return
        self.camera_D = specs.D
        try:
            # The SDK calibration is stored against the native calibration
            # frame. This mirrors reachy_mini.media.camera_base.CameraBase.
            original_size = (
                CameraResolution.R3840x2592at30fps.value[0],
                CameraResolution.R3840x2592at30fps.value[1],
            )
            self.camera_K = scale_intrinsics(
                specs.K,
                original_size,
                (width, height),
                crop_scale=1.0,
            )
        except Exception:
            self.camera_K = specs.K

    def start(self) -> bool:
        """Open the camera and start the background capture thread."""
        if not self.open():
            return False
        self._alive = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        interval = 1.0 / self.capture_fps
        t_last = 0.0
        n_frames = 0
        t_start = time.monotonic()
        while self._alive:
            now = time.monotonic()
            sleep_for = interval - (now - t_last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            t_last = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                break
            with self._cap_lock:
                ret, frame = self._cap.read()
            if not ret:
                continue
            t_frame = time.monotonic()
            jpg = self._encode_frame_bytes(frame)
            with self._lock:
                self._latest = (t_frame, frame)
                self._latest_jpeg = jpg
                if jpg is not None:
                    self._ring.append((t_frame, jpg))
            n_frames += 1
            elapsed = time.monotonic() - t_start
            if elapsed > 0:
                self._actual_fps = n_frames / elapsed

    def _encode_frame_bytes(self, frame: np.ndarray) -> Optional[bytes]:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        return jpg.tobytes()

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        jpg = self._encode_frame_bytes(frame)
        if jpg is None:
            return None
        return base64.b64encode(jpg).decode("ascii")

    @staticmethod
    def _base64_jpeg(jpg: bytes) -> str:
        return base64.b64encode(jpg).decode("ascii")

    def get_speech_frames(
        self,
        speech_start: float,
        speech_end: float,
        max_frames: int = 3,
    ) -> list[str]:
        """Return frames from the speech window [speech_start, speech_end].

        When max_frames == 1, returns the most recent frame (best context).
        When max_frames > 1, evenly samples across the window including
        PRE_SPEECH_SECS lookback for temporal context.
        """
        if max_frames > 1:
            window_start = speech_start - PRE_SPEECH_SECS
        else:
            window_start = speech_start

        with self._lock:
            candidates = [(t, jpg) for t, jpg in self._ring
                          if window_start <= t <= speech_end]

        if not candidates:
            with self._lock:
                if self._ring:
                    candidates = [(self._ring[-1][0], self._ring[-1][1])]
                elif self._latest_jpeg is not None:
                    candidates = [(speech_end, self._latest_jpeg)]
                else:
                    return []

        if max_frames == 1:
            selected = [candidates[-1][1]]
        elif len(candidates) <= max_frames:
            selected = [jpg for _, jpg in candidates]
        else:
            step = len(candidates) / max_frames
            selected = [candidates[int(i * step)][1] for i in range(max_frames)]

        return [self._base64_jpeg(jpg) for jpg in selected]

    def capture_single(self) -> Optional[str]:
        """Grab the latest frame from the ring buffer."""
        with self._lock:
            jpg = self._latest_jpeg
        return self._base64_jpeg(jpg) if jpg is not None else None

    def latest_raw(self, *, copy: bool = True) -> Optional[np.ndarray]:
        """Return the newest raw BGR frame captured by the background thread."""
        with self._lock:
            if self._latest is None:
                return None
            _, frame = self._latest
            return frame.copy() if copy else frame

    def read_live(self) -> Optional[str]:
        """Encode the latest buffered frame for live UI/VLM consumers."""
        with self._lock:
            jpg = self._latest_jpeg
        return self._base64_jpeg(jpg) if jpg is not None else None

    def read_raw_live(self) -> Optional[np.ndarray]:
        """Return the latest raw BGR frame.

        Kept for compatibility with existing tracking/diagnostic callers.
        Hardware reads are owned by the background capture thread.
        """
        return self.latest_raw()

    @property
    def buffer_count(self) -> int:
        with self._lock:
            return len(self._ring)

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    def health_check(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and self._alive

    def close(self):
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.camera_specs = None
        self.camera_K = None
        self.camera_D = None
        self._ring.clear()
        self._latest = None
        self._latest_jpeg = None
