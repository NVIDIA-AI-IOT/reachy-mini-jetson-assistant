# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-writer 100 Hz motion controller for tracking and speaking gestures.

Face tracking owns the persistent body/head targets.  A short official Pollen
recording can be layered on top while Reachy speaks; tracking remains the
priority and continues to follow the person throughout the gesture.
"""

from collections import deque
import threading
import time
from typing import Optional

import numpy as np

from reachy_mini.motion.goto import GotoMove
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    compose_world_offset,
    delta_angle_between_mat_rot,
    linear_pose_interpolation,
)

_TICK_HZ = 100
_TICK_PERIOD = 1.0 / _TICK_HZ
_SMOOTHING = 0.38
_SEND_EPS_DEG = 0.25
_SETTLED_EPS_DEG = 0.40
_ERROR_LOG_PERIOD_SECS = 2.0
_GESTURE_ENTER_SECS = 0.45
_GESTURE_LEAVE_SECS = 0.65
_FINAL_BODY_LIMIT_RAD = np.deg2rad(50.0)

_NEUTRAL_HEAD = np.eye(4, dtype=np.float64)
_NEUTRAL_ANTENNAS = np.array([0.0, 0.0], dtype=np.float64)


class MovementManager:
    """Blend live face tracking with an optional queued speaking gesture."""

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

        # Face-tracking targets and smoothed current values, in degrees.
        self._t_body = 0.0
        self._t_pitch = 0.0
        self._t_yaw = 0.0
        self._t_head_pose: Optional[np.ndarray] = None
        self._hold_motion = False
        self._target_version = 0
        self._c_body = 0.0
        self._c_pitch = 0.0
        self._c_yaw = 0.0
        self._c_head_pose: Optional[np.ndarray] = None

        # Primary recorded motion. Tracking is composed on top every tick.
        self._gesture_queue: deque[Move] = deque()
        self._gesture_current: Optional[Move] = None
        self._gesture_start = 0.0
        self._gesture_head = _NEUTRAL_HEAD.copy()
        self._gesture_antennas = _NEUTRAL_ANTENNAS.copy()
        self._gesture_body = 0.0

        self._last_sent: Optional[tuple[float, float, float]] = None
        self._last_sent_head_pose: Optional[np.ndarray] = None
        self._last_sent_version = -1
        self._last_error: Optional[str] = None
        self._last_error_log_time = 0.0
        self._last_pose_log_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="motion-100hz")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

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
        """Set an SDK-computed 4x4 target when no gesture is playing."""
        with self._lock:
            if self._t_head_pose is None or not np.allclose(head_pose, self._t_head_pose, atol=1e-4):
                self._target_version += 1
            self._t_head_pose = head_pose.copy()
            self._hold_motion = False

    def hold_current(self) -> None:
        """Hold the tracking layer; a speaking gesture may still animate."""
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
        self.stop_gesture()
        self.set_targets(0.0, 0.0, 0.0)

    def play_gesture(self, recorded: Move) -> bool:
        """Replace any gesture with one smooth recorded-move sequence."""
        try:
            target_head, target_antennas, target_body = recorded.evaluate(0.0)
            end_head, end_antennas, end_body = recorded.evaluate(recorded.duration - 1e-4)
        except Exception as exc:
            self._record_error(f"gesture setup failed: {exc}")
            return False

        with self._lock:
            enter = GotoMove(
                start_head_pose=self._gesture_head.copy(),
                target_head_pose=target_head,
                start_antennas=self._gesture_antennas.copy(),
                target_antennas=np.asarray(target_antennas, dtype=np.float64),
                start_body_yaw=self._gesture_body,
                target_body_yaw=float(target_body),
                duration=_GESTURE_ENTER_SECS,
                method=InterpolationTechnique.MIN_JERK,
            )
            leave = GotoMove(
                start_head_pose=end_head,
                target_head_pose=_NEUTRAL_HEAD.copy(),
                start_antennas=np.asarray(end_antennas, dtype=np.float64),
                target_antennas=_NEUTRAL_ANTENNAS.copy(),
                start_body_yaw=float(end_body),
                target_body_yaw=0.0,
                duration=_GESTURE_LEAVE_SECS,
                method=InterpolationTechnique.MIN_JERK,
            )
            self._gesture_queue.clear()
            self._gesture_current = None
            self._gesture_queue.extend((enter, recorded, leave))
        return True

    def stop_gesture(self) -> None:
        """Smoothly return the gesture layer to neutral."""
        with self._lock:
            self._gesture_queue.clear()
            already_neutral = (
                self._gesture_current is None
                and np.allclose(self._gesture_head, _NEUTRAL_HEAD, atol=1e-4)
                and np.allclose(self._gesture_antennas, _NEUTRAL_ANTENNAS, atol=1e-4)
                and abs(self._gesture_body) < 1e-4
            )
            if already_neutral:
                return
            self._gesture_current = GotoMove(
                start_head_pose=self._gesture_head.copy(),
                target_head_pose=_NEUTRAL_HEAD.copy(),
                start_antennas=self._gesture_antennas.copy(),
                target_antennas=_NEUTRAL_ANTENNAS.copy(),
                start_body_yaw=self._gesture_body,
                target_body_yaw=0.0,
                duration=_GESTURE_LEAVE_SECS,
                method=InterpolationTechnique.MIN_JERK,
            )
            self._gesture_start = time.monotonic()

    @property
    def is_gesturing(self) -> bool:
        with self._lock:
            return self._gesture_current is not None or bool(self._gesture_queue)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _loop(self) -> None:
        while self._running:
            t_start = time.monotonic()
            self._tick()
            sleep_time = _TICK_PERIOD - (time.monotonic() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _advance_gesture_locked(self, now: float) -> bool:
        if self._gesture_current is not None:
            if now - self._gesture_start >= self._gesture_current.duration:
                self._gesture_current = None

        if self._gesture_current is None and self._gesture_queue:
            self._gesture_current = self._gesture_queue.popleft()
            self._gesture_start = now

        if self._gesture_current is None:
            return False

        t = min(now - self._gesture_start, self._gesture_current.duration - 1e-6)
        head, antennas, body = self._gesture_current.evaluate(max(0.0, t))
        if head is not None:
            self._gesture_head = head
        if antennas is not None:
            self._gesture_antennas = np.asarray(antennas, dtype=np.float64)
        if body is not None:
            self._gesture_body = float(body)
        return True

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            gesture_active = self._advance_gesture_locked(now)
            t_body, t_pitch, t_yaw = self._t_body, self._t_pitch, self._t_yaw
            t_head_pose = None if self._t_head_pose is None else self._t_head_pose.copy()
            hold_motion = self._hold_motion
            target_version = self._target_version

            if not hold_motion and t_head_pose is None:
                target_delta = max(
                    abs(t_body - self._c_body),
                    abs(t_pitch - self._c_pitch),
                    abs(t_yaw - self._c_yaw),
                )
                if target_delta < _SETTLED_EPS_DEG:
                    self._c_body, self._c_pitch, self._c_yaw = t_body, t_pitch, t_yaw
                else:
                    self._c_body += (t_body - self._c_body) * _SMOOTHING
                    self._c_pitch += (t_pitch - self._c_pitch) * _SMOOTHING
                    self._c_yaw += (t_yaw - self._c_yaw) * _SMOOTHING

            c_body, c_pitch, c_yaw = self._c_body, self._c_pitch, self._c_yaw
            gesture_head = self._gesture_head.copy()
            gesture_antennas = self._gesture_antennas.copy()
            gesture_body = self._gesture_body

        if gesture_active:
            self._send_layered_gesture(
                gesture_head, gesture_antennas, gesture_body,
                c_body, c_pitch, c_yaw,
            )
            return

        if hold_motion:
            return

        if t_head_pose is not None:
            self._send_head_pose(t_head_pose, target_version)
            return

        pose = (c_body, c_pitch, c_yaw)
        target_delta = max(abs(t_body - c_body), abs(t_pitch - c_pitch), abs(t_yaw - c_yaw))
        if self._last_sent is not None:
            sent_delta = max(abs(a - b) for a, b in zip(pose, self._last_sent))
            if (
                target_version == self._last_sent_version
                and sent_delta < _SEND_EPS_DEG
                and target_delta < _SETTLED_EPS_DEG
            ):
                return

        head = create_head_pose(yaw=c_body + c_yaw, pitch=c_pitch, degrees=True)
        if self._send(head=head, antennas=_NEUTRAL_ANTENNAS, body_yaw=np.radians(c_body)):
            self._last_sent = pose
            self._last_sent_head_pose = None
            self._last_sent_version = target_version

    def _send_layered_gesture(
        self,
        gesture_head: np.ndarray,
        gesture_antennas: np.ndarray,
        gesture_body: float,
        tracking_body: float,
        tracking_pitch: float,
        tracking_yaw: float,
    ) -> None:
        # Give tracking priority near its limits by reducing, not removing,
        # the expressive layer. Near-center faces receive the full recording.
        tracking_load = max(abs(tracking_body) / 30.0, abs(tracking_yaw) / 20.0)
        gesture_scale = max(0.55, 1.0 - 0.45 * min(1.0, tracking_load))
        expressive_head = linear_pose_interpolation(_NEUTRAL_HEAD, gesture_head, gesture_scale)
        face_offset = create_head_pose(
            yaw=tracking_body + tracking_yaw,
            pitch=tracking_pitch,
            degrees=True,
        )
        final_head = compose_world_offset(expressive_head, face_offset)
        final_antennas = gesture_antennas * gesture_scale
        final_body = gesture_body * gesture_scale + np.radians(tracking_body)
        final_body = float(np.clip(final_body, -_FINAL_BODY_LIMIT_RAD, _FINAL_BODY_LIMIT_RAD))
        self._send(head=final_head, antennas=final_antennas, body_yaw=final_body)

    def _send_head_pose(self, target: np.ndarray, target_version: int) -> None:
        if self._c_head_pose is None:
            try:
                self._c_head_pose = self._reachy.get_current_head_pose()
            except Exception:
                self._c_head_pose = target.copy()

        angle_delta = delta_angle_between_mat_rot(self._c_head_pose[:3, :3], target[:3, :3])
        step_ratio = self._pose_smoothing
        if angle_delta > self._pose_max_step_rad:
            step_ratio = min(step_ratio, self._pose_max_step_rad / angle_delta)
        next_head_pose = linear_pose_interpolation(self._c_head_pose, target, step_ratio)

        if (
            target_version == self._last_sent_version
            and self._last_sent_head_pose is not None
            and np.allclose(next_head_pose, self._last_sent_head_pose, atol=1e-3)
        ):
            return
        if self._send(head=next_head_pose, antennas=_NEUTRAL_ANTENNAS):
            self._c_head_pose = next_head_pose
            self._last_sent_head_pose = next_head_pose
            self._last_sent_version = target_version

    def _send(self, **kwargs) -> bool:
        try:
            self._reachy.set_target(**kwargs)
            self._last_error = None
            return True
        except Exception as exc:
            self._record_error(str(exc))
            return False

    def _record_error(self, message: str) -> None:
        self._last_error = message
        now = time.monotonic()
        if now - self._last_error_log_time >= _ERROR_LOG_PERIOD_SECS:
            print(f"MovementManager: {message}")
            self._last_error_log_time = now
