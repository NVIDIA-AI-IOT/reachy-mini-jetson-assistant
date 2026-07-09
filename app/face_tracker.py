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

"""Face tracking — quickly acquire a usable face frame, then hold still.

Closed-loop visual servoing is used only while the face is poorly framed.
Each frame we measure how far the face is from the image center and step
the targets to reduce that error:

    head yaw   handles horizontal centering. A face on the right side of
               the image needs negative head yaw; this was measured with
               camera-in-the-loop calibration.
    body_yaw   provides large-range tracking and scan mode. When no face
               is visible, the robot sweeps body yaw across the room until
               a face is detected, then switches to tracking.
    head pitch handles vertical centering. On this physical unit a face above
               the image center needs negative pitch; a face below needs
               positive pitch.

Once the face is inside the good-frame zone, the robot locks its pose and
stops trying to perfect the framing. It only moves again when the face
drifts outside a wider reacquire zone. This keeps VLM captures stable and
prevents head/body hunting around detector jitter.
"""

import threading
import time
from typing import Optional

from app.camera import Camera
from app.face_detector import FaceBox, FaceDetector
from app.movement_manager import MovementManager

# Per-frame correction = gain * normalized_error. The loop is closed
# (re-measured every frame) so these only set responsiveness, not a
# calibrated scale. Horizontal tracking parameters are configurable so
# tuning can happen in settings.yaml without code edits.
_PITCH_GAIN_DEG = 5.0     # head pitch degrees per unit vertical error
_PITCH_STEP_MAX = 1.2     # max pitch degrees per frame
_PITCH_MAX = 18.0         # head pitch travel limit (deg)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class FaceTracker:
    """Background thread that frames the largest face, then holds pose.

    Args:
        camera:    frame source (raw BGR reads).
        detector:  YuNet face detector.
        manager:   100 Hz motion controller.
        reachy:    ReachyMini (unused for control, kept for API symmetry).
        fps:       detection/control rate.
        dead_zone: normalized error below which the axis is near center.
        good_frame_zone: normalized error that is good enough for capture.
        min_face_size: minimum face width/height fraction for stable capture.
        stable_frames: consecutive good frames required before capture is stable.
        face_lost_delay: seconds to hold before easing back to neutral
                         (only used when return_to_neutral is True).
        body_max_deg:    optional body_yaw travel limit (degrees).
        invert_body:     flip body_yaw direction. Enabled by default for
                         this robot because measured behavior showed that
                         a right-edge face needs negative body_yaw.
        body_enabled:    enable body-yaw assist for very large horizontal
                         errors. Off by default; head yaw is primary.
        vertical:        also track vertically with bounded head pitch.
        return_to_neutral: after a sustained face loss, ease back to the
                         neutral pose. Off by default so the robot simply
                         holds its last heading (no jarring snap-back).
        scan_enabled:    sweep head yaw when no face is visible.
        scan_body_range_deg: max body yaw used for wide scan assist.
        scan_speed_deg_per_sec: head scan speed.
    """

    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        manager: MovementManager,
        reachy=None,
        fps: float = 5.0,
        dead_zone: float = 0.12,
        lock_zone: float = 0.55,
        reacquire_zone: float = 0.85,
        good_frame_zone: Optional[float] = None,
        min_face_size: float = 0.06,
        stable_frames: int = 2,
        face_lost_delay: float = 3.0,
        head_yaw_max_deg: float = 20.0,
        head_yaw_gain: float = 18.0,
        head_yaw_step: float = 1.0,
        soft_center_head_yaw_max_deg: float = 12.0,
        soft_center_head_yaw_step: float = 0.5,
        body_max_deg: float = 30.0,
        body_gain: float = 12.0,
        body_step: float = 0.75,
        invert_body: bool = True,
        body_enabled: bool = False,
        vertical: bool = True,
        return_to_neutral: bool = False,
        scan_enabled: bool = True,
        scan_body_range_deg: float = 90.0,
        scan_speed_deg_per_sec: float = 35.0,
    ):
        self._camera = camera
        self._detector = detector
        self._manager = manager
        self._reachy = reachy
        self._period = 1.0 / fps
        self._dead_zone = dead_zone
        self._good_frame_zone = max(good_frame_zone if good_frame_zone is not None else lock_zone, dead_zone)
        self._lock_zone = self._good_frame_zone
        self._reacquire_zone = max(reacquire_zone, self._lock_zone)
        self._min_face_size = max(0.0, min_face_size)
        self._stable_frames_required = max(1, stable_frames)
        self._face_lost_delay = face_lost_delay
        self._head_yaw_max = head_yaw_max_deg
        self._head_yaw_gain = head_yaw_gain
        self._head_yaw_step = head_yaw_step
        self._soft_center_head_yaw_max = soft_center_head_yaw_max_deg
        self._soft_center_head_yaw_step = soft_center_head_yaw_step
        self._body_max = body_max_deg
        self._body_gain = body_gain
        self._body_step = body_step
        self._body_enabled = body_enabled
        self._vertical = vertical
        self._return_to_neutral = return_to_neutral
        self._scan_enabled = scan_enabled
        self._scan_body_range = scan_body_range_deg
        self._scan_speed = scan_speed_deg_per_sec
        # With invert_body=True, a face to the right (err_x > 0) commands
        # negative body_yaw. That matches the measured behavior on this unit.
        self._body_sign = 1.0 if not invert_body else -1.0

        self._enabled = True
        self._motion_frozen = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Targets (degrees). Horizontal tracking accumulates small bounded
        # corrections. Each detector update closes more of the remaining image
        # error until the face enters the good-frame zone.
        self._body = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._err_x = 0.0
        self._err_y = 0.0

        self._last_face_time: Optional[float] = None
        self._tracking_active = False
        self._pose_locked = False
        self._reacquiring = False
        self._reacquiring_x = False
        self._reacquiring_y = False
        self._scanning = False
        self._scan_direction = 1.0
        self._scan_body_direction = 1.0

        self._state_lock = threading.Lock()
        self._last_face_box: Optional[FaceBox] = None
        self._face_detected = False
        self._centered = False
        self._frame_good = False
        self._stable = False
        self._stable_count = 0
        self._last_debug_log_time = 0.0

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="face-tracker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._reset()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._reset()

    def set_motion_frozen(self, frozen: bool) -> None:
        """Freeze motor commands while continuing face detection/state updates."""
        self._motion_frozen = frozen

    # ── status (thread-safe) ──────────────────────────────────────

    @property
    def is_tracking(self) -> bool:
        return self._tracking_active

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def face_detected(self) -> bool:
        with self._state_lock:
            return self._face_detected

    @property
    def centered(self) -> bool:
        with self._state_lock:
            return self._centered

    @property
    def frame_good(self) -> bool:
        with self._state_lock:
            return self._frame_good

    @property
    def stable(self) -> bool:
        with self._state_lock:
            return self._stable

    @property
    def last_face_box(self) -> Optional[FaceBox]:
        with self._state_lock:
            return self._last_face_box

    @property
    def target_yaw_deg(self) -> float:
        return self._yaw

    @property
    def target_body_yaw_deg(self) -> float:
        return self._body

    @property
    def error_x(self) -> float:
        return self._err_x

    @property
    def pose_locked(self) -> bool:
        return self._pose_locked

    def wait_until_stable(self, timeout: float, poll_interval: float = 0.03) -> bool:
        """Wait briefly for a good, stable face frame without blocking forever."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.stable:
                return True
            time.sleep(poll_interval)
        return self.stable

    # ── main loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            t_start = time.monotonic()
            if self._enabled:
                self._tick()
            sleep_time = self._period - (time.monotonic() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self) -> None:
        frame = self._camera.read_raw_live()
        if frame is None:
            self._tracking_active = False
            return

        box = self._detector.detect(frame)
        now = time.monotonic()

        if box is not None:
            self._last_face_time = now
            self._tracking_active = True
            self._scanning = False
            self._servo(box, frame.shape, apply_motion=not self._motion_frozen)
            return

        with self._state_lock:
            self._face_detected = False
            self._last_face_box = None
            self._centered = False
            self._frame_good = False
            self._stable = False
            self._stable_count = 0
        self._tracking_active = False
        if not self._motion_frozen:
            self._handle_face_lost(now)

    def _servo(self, box: FaceBox, frame_shape, *, apply_motion: bool = True) -> None:
        h, w = frame_shape[:2]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        face_w = max(0.0, box[2] - box[0])
        face_h = max(0.0, box[3] - box[1])
        face_size = max(face_w / max(w, 1), face_h / max(h, 1))
        err_x = (cx - w / 2.0) / (w / 2.0)   # +1 = face at right edge
        err_y = (cy - h / 2.0) / (h / 2.0)   # +1 = face at bottom edge
        self._err_x = err_x
        self._err_y = err_y

        centered_x = abs(err_x) < self._dead_zone
        centered_y = abs(err_y) < self._dead_zone
        good_x = abs(err_x) < self._good_frame_zone
        good_y = abs(err_y) < self._good_frame_zone
        face_large_enough = face_size >= self._min_face_size
        frame_good = face_large_enough and good_x and (good_y or not self._vertical)
        reacquire_x = abs(err_x) > self._reacquire_zone
        reacquire_y = self._vertical and abs(err_y) > self._reacquire_zone
        reacquire = reacquire_x or reacquire_y

        action_needed = face_large_enough and not frame_good

        # Hysteresis: once a far-edge face starts wide reacquisition, keep
        # head/body assistance active until the face reaches the good-frame
        # zone. Dropping back at the reacquire threshold can strand the face
        # between the 0.18 good zone and the 0.45 entry threshold.
        if reacquire_x:
            self._reacquiring_x = True
        elif good_x:
            self._reacquiring_x = False

        if self._vertical:
            if reacquire_y:
                self._reacquiring_y = True
            elif good_y:
                self._reacquiring_y = False
        else:
            self._reacquiring_y = False

        self._reacquiring = self._reacquiring_x or self._reacquiring_y

        if self._pose_locked and self._reacquiring:
            self._pose_locked = False

        rebalanced = False
        if not action_needed:
            apply_motion = False
            self._pose_locked = True
            if not self._motion_frozen:
                rebalanced = self._rebalance_head_into_body()
                if not rebalanced:
                    self._manager.hold_current()

        # Update each axis independently, then publish one coherent target.
        # A far vertical error must not grant extra yaw/body authority when
        # horizontal framing is already good.
        if apply_motion and action_needed:
            actions = []

            if not good_x:
                yaw_limit = (
                    self._head_yaw_max
                    if self._reacquiring_x
                    else min(self._head_yaw_max, self._soft_center_head_yaw_max)
                )
                yaw_step = (
                    self._head_yaw_step
                    if self._reacquiring_x
                    else min(self._head_yaw_step, self._soft_center_head_yaw_step)
                )
                yaw_delta = _clamp(-self._head_yaw_gain * err_x, -yaw_step, yaw_step)
                self._yaw = _clamp(self._yaw + yaw_delta, -yaw_limit, yaw_limit)
                actions.append("yaw")

                if self._reacquiring_x and self._body_enabled:
                    # Keep turning while the face remains near a horizontal
                    # edge. Vertical displacement alone never rotates body.
                    body_delta = _clamp(
                        self._body_sign * self._body_gain * err_x,
                        -self._body_step,
                        self._body_step,
                    )
                    self._body = _clamp(
                        self._body + body_delta,
                        -self._body_max,
                        self._body_max,
                    )
                    actions.append("body")

            if self._vertical and not good_y:
                # Camera-in-the-loop calibration on this unit: negative head
                # pitch looks upward. err_y is negative above image center,
                # so it is applied directly rather than inverted.
                pitch_delta = _clamp(
                    _PITCH_GAIN_DEG * err_y,
                    -_PITCH_STEP_MAX,
                    _PITCH_STEP_MAX,
                )
                self._pitch = _clamp(
                    self._pitch + pitch_delta,
                    -_PITCH_MAX,
                    _PITCH_MAX,
                )
                actions.append("pitch")

            self._manager.set_targets(self._body, self._pitch, self._yaw)
            self._log_tracking(
                err_x,
                err_y,
                face_size,
                frame_good,
                reacquire,
                "+".join(actions) if actions else "hold",
            )
        elif rebalanced:
            self._log_tracking(
                err_x, err_y, face_size, frame_good, reacquire, "rebalance",
            )
        elif box is not None:
            self._log_tracking(
                err_x, err_y, face_size, frame_good, reacquire, "hold",
            )
        # If the face is already in the good-frame zone, hold the current
        # pose. Relaxing back to neutral here causes visible micro-motion and
        # blurs frames captured for the VLM.

        if frame_good:
            # Once reasonably framed, lock the pose. This keeps the camera
            # stable for human viewing and VLM captures, and prevents
            # chasing detector jitter or trying to over-perfect centering.
            self._pose_locked = True
            self._stable_count += 1
        else:
            self._stable_count = 0

        with self._state_lock:
            self._face_detected = True
            self._last_face_box = box
            self._frame_good = frame_good
            self._stable = self._stable_count >= self._stable_frames_required
            self._centered = frame_good or (centered_x and (centered_y or not self._vertical))

    def _rebalance_head_into_body(self) -> bool:
        """Move yaw trim into body yaw without changing camera direction.

        Head world yaw is body + yaw in MovementManager. Applying the same
        signed amount to body and removing it from head yaw preserves that
        sum, so a centered face stays centered while the head returns toward
        a natural forward pose.
        """
        if not self._body_enabled or abs(self._yaw) < 1e-6:
            return False

        transfer = _clamp(self._yaw, -self._body_step, self._body_step)
        next_body = _clamp(
            self._body + transfer,
            -self._body_max,
            self._body_max,
        )
        applied = next_body - self._body
        if abs(applied) < 1e-6:
            return False

        self._body = next_body
        self._yaw -= applied
        self._manager.set_targets(self._body, self._pitch, self._yaw)
        return True

    def _log_tracking(
        self,
        err_x: float,
        err_y: float,
        face_size: float,
        frame_good: bool,
        reacquire: bool,
        action: str,
    ) -> None:
        now = time.monotonic()
        if now - self._last_debug_log_time < 1.0:
            return
        self._last_debug_log_time = now
        print(
            "FaceTracker: "
            f"action={action} err=({err_x:+.2f},{err_y:+.2f}) "
            f"face_size={face_size:.2f} good={frame_good} "
            f"reacquire={reacquire} locked={self._pose_locked} "
            f"target=(body={self._body:+.1f},pitch={self._pitch:+.1f},yaw={self._yaw:+.1f})deg"
        )

    def _handle_face_lost(self, now: float) -> None:
        # If enabled, scan horizontally until a face enters view. On brief
        # detector dropouts we hold the last heading for face_lost_delay so
        # the robot does not immediately sweep away from the person.
        if self._scan_enabled:
            if self._last_face_time is None or now - self._last_face_time >= self._face_lost_delay:
                self._pose_locked = False
                self._reacquiring = False
                self._reacquiring_x = False
                self._reacquiring_y = False
                self._scan_for_face()
            return

        # Otherwise hold by default; only return to neutral if explicitly
        # enabled, and then via a slow decay rather than a snap.
        if not self._tracking_active or self._last_face_time is None:
            return
        if not self._return_to_neutral:
            return
        if now - self._last_face_time < self._face_lost_delay:
            return
        self._body *= 0.97
        self._pitch *= 0.97
        self._yaw *= 0.97
        self._manager.set_targets(self._body, self._pitch, self._yaw)
        if abs(self._body) < 0.5 and abs(self._pitch) < 0.5 and abs(self._yaw) < 0.5:
            self._reset()
            self._tracking_active = False
            self._last_face_time = None

    def _scan_for_face(self) -> None:
        self._scanning = True
        self._tracking_active = False

        # Sweep the head/camera first. Body yaw alone can leave the camera
        # looking at nearly the same field of view depending on SDK pose
        # composition, so scan must visibly move head yaw. Body yaw pans
        # slowly underneath it for wider coverage.
        head_limit = self._head_yaw_max
        step = self._scan_speed * self._period * self._scan_direction
        next_yaw = self._yaw + step

        if next_yaw >= head_limit:
            next_yaw = head_limit
            self._scan_direction = -1.0
        elif next_yaw <= -head_limit:
            next_yaw = -head_limit
            self._scan_direction = 1.0

        if self._body_enabled:
            scan_limit = min(self._scan_body_range, self._body_max)
            body_step = self._scan_speed * 0.35 * self._period * self._scan_body_direction
            next_body = self._body + body_step
            if next_body >= scan_limit:
                next_body = scan_limit
                self._scan_body_direction = -1.0
            elif next_body <= -scan_limit:
                next_body = -scan_limit
                self._scan_body_direction = 1.0
            self._body = next_body

        self._yaw = next_yaw
        # Preserve the last vertical angle during horizontal search. Resetting
        # pitch here makes a briefly lost high/low face trigger a down/up bob,
        # then repeat the same failed reacquisition cycle.
        self._manager.set_targets(self._body, self._pitch, self._yaw)

    def _reset(self) -> None:
        self._body = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._pose_locked = False
        self._reacquiring = False
        self._reacquiring_x = False
        self._reacquiring_y = False
        self._scanning = False
        self._stable_count = 0
        with self._state_lock:
            self._face_detected = False
            self._last_face_box = None
            self._centered = False
            self._frame_good = False
            self._stable = False
        self._manager.reset()
