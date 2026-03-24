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

"""Emotion detection — vision-based face emotion recognition.

Two-stage pipeline:
  1. YuNet (MIT, OpenCV Zoo) — face detection via cv2.FaceDetectorYN
  2. FER+ int8 (MIT, ONNX Model Zoo) — emotion classification via ONNX Runtime

Reacts to facial expressions captured by the camera, with text-based
greeting/farewell detection as a fallback.
"""

import base64
import re
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "emotion"

_FACE_MODEL_URL = (
    "https://huggingface.co/opencv/face_detection_yunet"
    "/resolve/main/face_detection_yunet_2023mar.onnx"
)
_EMOTION_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8"
    "/resolve/main/emotion-ferplus-12-int8.onnx"
)

_FACE_MODEL_FILE = "yunet_2023mar.onnx"
_EMOTION_MODEL_FILE = "emotion-ferplus-12-int8.onnx"

_YUNET_SCORE_THRESH = 0.5
_YUNET_NMS_THRESH = 0.3
_YUNET_TOP_K = 5000

_FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]


class Emotion(Enum):
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    DISGUSTED = "disgusted"
    SCARED = "scared"
    CONTEMPT = "contempt"
    GREETING = "greeting"
    FAREWELL = "farewell"
    NEUTRAL = "neutral"


_FERPLUS_TO_EMOTION = {
    "neutral": Emotion.NEUTRAL,
    "happiness": Emotion.HAPPY,
    "surprise": Emotion.SURPRISED,
    "sadness": Emotion.SAD,
    "anger": Emotion.ANGRY,
    "disgust": Emotion.DISGUSTED,
    "fear": Emotion.SCARED,
    "contempt": Emotion.CONTEMPT,
}


@dataclass
class EmotionResult:
    emotion: Emotion
    confidence: float
    inference_ms: float
    face_detected: bool = False
    face_box: Optional[tuple[int, int, int, int]] = None


_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up|yo)\b",
    re.I,
)
_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|see\s+you|later|good\s*night|take\s+care)\b", re.I
)


# ── Model download ───────────────────────────────────────────────


def _download_file(url: str, path: Path, label: str) -> bool:
    try:
        import httpx
    except ImportError:
        print("Emotion: install httpx to auto-download model (pip install httpx)")
        return False
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=262144):
                    f.write(chunk)
                    done += len(chunk)
                    if total and total > 0:
                        pct = 100 * done / total
                        sys.stdout.write(f"\r  {label}: {pct:.0f}%\r")
                        sys.stdout.flush()
        if total:
            print()
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if path.exists():
            path.unlink()
        return False


def _ensure_model_files() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    needed: list[tuple[str, Path, str]] = []

    face_path = MODELS_DIR / _FACE_MODEL_FILE
    if not face_path.exists():
        needed.append((_FACE_MODEL_URL, face_path, "YuNet face (~233 KB)"))

    emo_path = MODELS_DIR / _EMOTION_MODEL_FILE
    if not emo_path.exists():
        needed.append((_EMOTION_MODEL_URL, emo_path, "FER+ int8 (~19 MB)"))

    if not needed:
        return True

    print("Downloading emotion models...")
    for url, path, label in needed:
        if not _download_file(url, path, label):
            return False
        print(f"  Saved {path}")
    return True


# ── Provider selection (for FER+ ONNX Runtime) ───────────────────


def _pick_providers(ort) -> list[str]:
    """Prefer GPU providers if available, fall back to CPU."""
    available = ort.get_available_providers()
    preferred = []
    if "TensorrtExecutionProvider" in available:
        preferred.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")
    return preferred


# ── YuNet face detection helpers ─────────────────────────────────


def _pick_best_face(faces: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Select the highest-confidence face from YuNet output.

    YuNet returns Nx15 array per face:
      [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]

    Returns (x1, y1, x2, y2) or None.
    """
    if faces is None or len(faces) == 0:
        return None
    best = faces[faces[:, -1].argmax()]
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    if w < 10 or h < 10:
        return None
    return (x, y, x + w, y + h)


# ── EmotionDetector ──────────────────────────────────────────────


class EmotionDetector:
    """Vision-based face emotion detector.

    Stage 1: YuNet (MIT, OpenCV Zoo) — face detection via cv2.FaceDetectorYN.
    Stage 2: FER+ int8 (MIT, ONNX Model Zoo) — emotion classification via ONNX Runtime.

    YuNet runs on CPU via OpenCV DNN (~2 ms for 233 KB model).
    FER+ runs on GPU (CUDA/TensorRT) or CPU via ONNX Runtime.
    """

    def __init__(self):
        self._face_detector: Optional[cv2.FaceDetectorYN] = None
        self._face_lock = threading.Lock()
        self._emo_session = None
        self._provider = "CPUExecutionProvider"

    def load(self) -> bool:
        if not _ensure_model_files():
            return False
        try:
            face_model_path = str(MODELS_DIR / _FACE_MODEL_FILE)
            self._face_detector = cv2.FaceDetectorYN.create(
                face_model_path, "",
                (320, 320),
                score_threshold=_YUNET_SCORE_THRESH,
                nms_threshold=_YUNET_NMS_THRESH,
                top_k=_YUNET_TOP_K,
            )

            import onnxruntime as ort

            providers = _pick_providers(ort)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._emo_session = ort.InferenceSession(
                str(MODELS_DIR / _EMOTION_MODEL_FILE),
                sess_options=opts,
                providers=providers,
            )
            self._provider = self._emo_session.get_providers()[0]
            return True
        except Exception as e:
            print(f"Emotion: load error — {e}")
            return False

    @property
    def provider(self) -> str:
        return self._provider

    def detect_face(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Run YuNet face detection on a raw BGR frame (no emotion classification).

        Thread-safe. Returns (x1, y1, x2, y2) or None.
        Used by FaceTracker for continuous head tracking.
        """
        if self._face_detector is None:
            return None
        h, w = frame.shape[:2]
        with self._face_lock:
            self._face_detector.setInputSize((w, h))
            _, faces = self._face_detector.detect(frame)
        return _pick_best_face(faces)

    def detect(self, frame_b64: str) -> EmotionResult:
        """Classify emotion from a base64-encoded JPEG camera frame."""
        if not frame_b64:
            return EmotionResult(Emotion.NEUTRAL, 0.0, 0.0)

        t0 = time.perf_counter()

        jpg_bytes = base64.b64decode(frame_b64)
        img = cv2.imdecode(
            np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            return EmotionResult(Emotion.NEUTRAL, 0.0, 0.0)

        box = self.detect_face(img)

        if box is None:
            dt = (time.perf_counter() - t0) * 1000
            return EmotionResult(Emotion.NEUTRAL, 0.0, dt, face_detected=False)

        x1, y1, x2, y2 = box

        face_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        tensor = gray.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, np.newaxis, :, :]  # [1, 1, 64, 64]

        emo_input_name = self._emo_session.get_inputs()[0].name
        emo_out = self._emo_session.run(None, {emo_input_name: tensor})
        probs = _softmax(emo_out[0][0])

        idx = int(np.argmax(probs))
        label = _FERPLUS_LABELS[idx]
        emotion = _FERPLUS_TO_EMOTION[label]
        confidence = float(probs[idx])

        dt = (time.perf_counter() - t0) * 1000
        return EmotionResult(
            emotion, confidence, dt,
            face_detected=True, face_box=(x1, y1, x2, y2),
        )

    def detect_text(self, text: str) -> Optional[Emotion]:
        """Check for greeting/farewell via text patterns (no ML model)."""
        if not text or not text.strip():
            return None
        if _GREETING_RE.search(text):
            return Emotion.GREETING
        if _FAREWELL_RE.search(text):
            return Emotion.FAREWELL
        return None

    def health_check(self) -> bool:
        return self._face_detector is not None and self._emo_session is not None

    def unload(self):
        self._face_detector = None
        if self._emo_session:
            del self._emo_session
            self._emo_session = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
