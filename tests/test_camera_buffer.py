# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import time
import unittest

import cv2
import numpy as np

from app.camera import Camera, resolve_camera_device


def test_camera_auto_discovery_uses_reachy_video_index_zero(tmp_path):
    target = tmp_path / "video3"
    target.touch()
    link = tmp_path / "usb-SunplusIT_Inc_Reachy_Mini_Camera-unit-video-index0"
    link.symlink_to(target)

    assert resolve_camera_device(-1, tmp_path) == 3


def test_camera_explicit_device_is_preserved(tmp_path):
    assert resolve_camera_device(5, tmp_path) == 5


class CameraBufferTests(unittest.TestCase):
    def test_ring_keeps_compressed_jpeg_bytes(self):
        camera = Camera(width=1920, height=1080, capture_fps=10.0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        jpg = camera._encode_frame_bytes(frame)

        self.assertIsNotNone(jpg)
        assert jpg is not None
        camera._ring.append((time.monotonic(), jpg))

        self.assertIsInstance(camera._ring[-1][1], bytes)
        self.assertLess(len(camera._ring[-1][1]), frame.nbytes)

    def test_speech_capture_returns_decodable_jpeg(self):
        camera = Camera(capture_fps=10.0)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        jpg = camera._encode_frame_bytes(frame)
        self.assertIsNotNone(jpg)
        assert jpg is not None

        timestamp = time.monotonic()
        camera._ring.append((timestamp, jpg))
        camera._latest = (timestamp, frame)
        camera._latest_jpeg = jpg

        encoded = camera.get_speech_frames(timestamp - 0.1, timestamp + 0.1, max_frames=1)

        self.assertEqual(len(encoded), 1)
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded[0]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, frame.shape)
        self.assertEqual(camera.capture_single(), encoded[0])
        self.assertEqual(camera.read_live(), encoded[0])


if __name__ == "__main__":
    unittest.main()
