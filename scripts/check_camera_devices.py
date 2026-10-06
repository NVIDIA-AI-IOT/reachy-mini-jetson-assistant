#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe Reachy camera nodes through OpenCV by index and device path."""

import cv2


def main() -> int:
    print("OpenCV", cv2.__version__)
    print("V4L2 backend available:", cv2.videoio_registry.hasBackend(cv2.CAP_V4L2))
    opened = False
    for source in (0, 1, "/dev/video0", "/dev/video1"):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        ok = capture.isOpened()
        grabbed, frame = capture.read() if ok else (False, None)
        shape = tuple(frame.shape) if frame is not None else None
        print(f"source={source!r} opened={ok} read={grabbed} shape={shape}")
        opened = opened or (ok and grabbed)
        capture.release()
    return 0 if opened else 1


if __name__ == "__main__":
    raise SystemExit(main())
