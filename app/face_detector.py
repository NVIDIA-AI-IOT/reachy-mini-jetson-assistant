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

"""Face detection — YuNet (MIT, OpenCV Zoo) via cv2.FaceDetectorYN.

A single-purpose detector used to locate the most prominent face in a
camera frame. The face box drives head tracking so the robot keeps the
person centered in view. No emotion or expression classification.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_FACE_DIR = _MODELS_ROOT / "face"
_LEGACY_DIR = _MODELS_ROOT / "emotion"  # original location of the YuNet weights

_FACE_MODEL_FILE = "yunet_2023mar.onnx"
_FACE_MODEL_URL = (
    "https://huggingface.co/opencv/face_detection_yunet"
    "/resolve/main/face_detection_yunet_2023mar.onnx"
)

_SCORE_THRESH = 0.5
_NMS_THRESH = 0.3
_TOP_K = 5000
_MIN_FACE_PX = 10

# YuNet on full 1080p frames is slow on CPU and adds control-loop lag.
# Detect on a downscaled copy and scale the box back to original coords.
_DETECT_MAX_W = 640

FaceBox = tuple[int, int, int, int]


def _resolve_model_path() -> Optional[Path]:
    """Return an existing YuNet weights path, downloading it if needed."""
    for candidate in (_FACE_DIR / _FACE_MODEL_FILE, _LEGACY_DIR / _FACE_MODEL_FILE):
        if candidate.exists():
            return candidate
    dest = _FACE_DIR / _FACE_MODEL_FILE
    if _download(_FACE_MODEL_URL, dest, "YuNet face (~233 KB)"):
        return dest
    return None


def _download(url: str, path: Path, label: str) -> bool:
    try:
        import httpx
    except ImportError:
        print("FaceDetector: install httpx to auto-download model (pip install httpx)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=262144):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r  {label}: {100 * done / total:.0f}%")
                        sys.stdout.flush()
        if total:
            print()
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if path.exists():
            path.unlink()
        return False


def _pick_best_face(faces: Optional[np.ndarray]) -> Optional[FaceBox]:
    """Return the highest-confidence face as (x1, y1, x2, y2), or None.

    YuNet returns an Nx15 array per face where columns 0..3 are the
    bounding box (x, y, w, h) and the last column is the score.
    """
    if faces is None or len(faces) == 0:
        return None
    best = faces[faces[:, -1].argmax()]
    x, y, w, h = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
    if w < _MIN_FACE_PX or h < _MIN_FACE_PX:
        return None
    return (x, y, x + w, y + h)


class FaceDetector:
    """Thread-safe YuNet face detector (CPU, via OpenCV)."""

    def __init__(self):
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        model_path = _resolve_model_path()
        if model_path is None:
            return False
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path), "",
                (320, 320),
                score_threshold=_SCORE_THRESH,
                nms_threshold=_NMS_THRESH,
                top_k=_TOP_K,
            )
            return True
        except Exception as e:
            print(f"FaceDetector: load error — {e}")
            return False

    @property
    def backend(self) -> str:
        return "YuNet (OpenCV CPU)"

    def detect(self, frame: np.ndarray) -> Optional[FaceBox]:
        """Detect the most prominent face in a raw BGR frame.

        Returns (x1, y1, x2, y2) in the original frame's pixel coordinates,
        or None. Detection runs on a downscaled copy for speed. Thread-safe.
        """
        if self._detector is None or frame is None:
            return None
        h, w = frame.shape[:2]

        scale = _DETECT_MAX_W / w if w > _DETECT_MAX_W else 1.0
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        sh, sw = small.shape[:2]

        with self._lock:
            self._detector.setInputSize((sw, sh))
            _, faces = self._detector.detect(small)

        box = _pick_best_face(faces)
        if box is None or scale >= 1.0:
            return box
        inv = 1.0 / scale
        return (int(box[0] * inv), int(box[1] * inv), int(box[2] * inv), int(box[3] * inv))

    def health_check(self) -> bool:
        return self._detector is not None

    def unload(self) -> None:
        self._detector = None
