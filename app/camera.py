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

"""Camera — USB webcam with background ring buffer for VLM context.

The background thread continuously grabs frames into a timestamped ring buffer.
When speech ends, get_speech_frames() returns evenly-sampled frames spanning
the full speech window (0.5s pre-speech through speech end).
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
        self._cap_lock = threading.Lock()
        self._ring: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(1, int(capture_fps * (MAX_SPEECH_SECS + PRE_SPEECH_SECS)))
        )
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
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        return True

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
            with self._lock:
                self._ring.append((time.monotonic(), frame))
            n_frames += 1
            elapsed = time.monotonic() - t_start
            if elapsed > 0:
                self._actual_fps = n_frames / elapsed

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        return base64.b64encode(jpg.tobytes()).decode("ascii")

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
            candidates = [(t, f) for t, f in self._ring
                          if window_start <= t <= speech_end]

        if not candidates:
            with self._lock:
                if self._ring:
                    candidates = [(self._ring[-1][0], self._ring[-1][1])]
                else:
                    return []

        if max_frames == 1:
            selected = [candidates[-1][1]]
        elif len(candidates) <= max_frames:
            selected = [f for _, f in candidates]
        else:
            step = len(candidates) / max_frames
            selected = [candidates[int(i * step)][1] for i in range(max_frames)]

        result = []
        for frame in selected:
            b64 = self._encode_frame(frame)
            if b64:
                result.append(b64)
        return result

    def capture_single(self) -> Optional[str]:
        """Grab the latest frame from the ring buffer."""
        with self._lock:
            if not self._ring:
                return None
            _, frame = self._ring[-1]
        return self._encode_frame(frame)

    def read_live(self) -> Optional[str]:
        """Read a fresh frame directly from the camera hardware.

        Unlike capture_single() (which reads from the 3fps ring buffer),
        this can be called at any rate for a smooth UI stream without
        affecting the VLM ring buffer.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        with self._cap_lock:
            ret, frame = self._cap.read()
        if not ret:
            return None
        return self._encode_frame(frame)

    def read_raw_live(self) -> Optional[np.ndarray]:
        """Read a fresh raw BGR frame directly from camera hardware.

        Returns the numpy array without JPEG encoding, for use by the
        face tracker and other consumers that need raw pixel access.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        with self._cap_lock:
            ret, frame = self._cap.read()
        return frame if ret else None

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
        self._ring.clear()
