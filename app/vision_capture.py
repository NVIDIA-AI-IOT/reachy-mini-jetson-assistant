"""Helpers for stable VLM image capture."""

import time
from typing import TYPE_CHECKING, Optional

from app.camera import Camera

if TYPE_CHECKING:
    from app.face_tracker import FaceTracker


def capture_frames_for_vlm(
    camera: Camera,
    tracker: Optional["FaceTracker"],
    n_frames: int,
    *,
    settle_secs: float,
    acquire_timeout_secs: float,
    inter_frame_secs: float = 0.08,
) -> tuple[list[str], bool]:
    """Freeze motion briefly and capture frames from the shared camera buffer.

    If tracking is active, wait only a bounded amount of time for a good face
    frame. The VLM interaction should not stall just because no face is usable.
    """
    stable_at_capture = False
    if tracker is not None and not tracker.stable:
        tracker.wait_until_stable(acquire_timeout_secs)

    if tracker is not None:
        tracker.set_motion_frozen(True)
        stable_at_capture = tracker.stable

    frames: list[str] = []
    try:
        if tracker is not None and settle_secs > 0:
            time.sleep(settle_secs)
            stable_at_capture = tracker.stable
        for _ in range(max(1, n_frames)):
            frame = camera.read_live()
            if frame:
                frames.append(frame)
            if n_frames > 1:
                time.sleep(inter_frame_secs)
    finally:
        if tracker is not None:
            tracker.set_motion_frozen(False)

    return frames, stable_at_capture
