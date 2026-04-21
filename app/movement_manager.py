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

"""100 Hz layered motion controller for Reachy Mini.

Adapted from Pollen Robotics' reachy_mini_conversation_app/moves.py.
Owns all robot motion through a single set_target() call per tick,
blending primary moves (emotions, breathing) with secondary face
tracking offsets via compose_world_offset().
"""

import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.motion.goto import GotoMove
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    compose_world_offset,
)

_TICK_HZ = 100
_TICK_PERIOD = 1.0 / _TICK_HZ
_IDLE_BEFORE_BREATHING = float("inf")

_NEUTRAL_HEAD = np.eye(4, dtype=np.float64)
_NEUTRAL_ANTENNAS = np.array([0.0, 0.0], dtype=np.float64)
_NEUTRAL_BODY_YAW = 0.0
_IDENTITY_4x4 = np.eye(4, dtype=np.float64)

_GOTO_NEUTRAL_DURATION = 1.0


class BreathingMove(Move):
    """Idle breathing animation — z-axis bobbing + antenna sway.

    Runs indefinitely until interrupted by a new queued move.
    At t=0 the pose is exactly neutral (sin(0)=0), so it blends
    seamlessly after a GotoMove to neutral.
    """

    Z_AMP_M = 0.005
    Z_FREQ = 0.1
    ANT_AMP_RAD = math.radians(15)
    ANT_FREQ = 0.5

    @property
    def duration(self) -> float:
        return float("inf")

    def evaluate(
        self, t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        z = self.Z_AMP_M * math.sin(2.0 * math.pi * self.Z_FREQ * t)
        head = create_head_pose(z=z * 1000, mm=True, degrees=True)

        ant = self.ANT_AMP_RAD * math.sin(2.0 * math.pi * self.ANT_FREQ * t)
        antennas = np.array([ant, -ant], dtype=np.float64)

        return head, antennas, 0.0


class MovementManager:
    """100 Hz motion controller blending primary moves with face tracking.

    Primary moves (emotion recordings, breathing) are queued and executed
    sequentially. Face tracking offsets are applied additively every tick
    via compose_world_offset so the robot can track faces *during*
    emotion animations.

    Public API (all thread-safe):
        queue_move(move)      — append to primary queue; interrupts breathing
        clear_queue()         — stop everything; idle timer restarts
        set_face_offsets(mat) — update secondary 4x4 head offset
        current_pose          — snapshot of last evaluated primary pose
        start() / stop()      — lifecycle
    """

    def __init__(self, reachy):
        self._reachy = reachy

        self._queue: deque[Move] = deque()
        self._current_move: Optional[Move] = None
        self._move_start: float = 0.0
        self._idle_since: float = 0.0
        self._is_breathing: bool = False

        self._last_head: npt.NDArray[np.float64] = _NEUTRAL_HEAD.copy()
        self._last_antennas: npt.NDArray[np.float64] = _NEUTRAL_ANTENNAS.copy()
        self._last_body_yaw: float = _NEUTRAL_BODY_YAW

        self._face_offsets: npt.NDArray[np.float64] = _IDENTITY_4x4.copy()
        self._face_offsets_dirty: bool = False
        self._face_lock = threading.Lock()

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sent_idle: bool = False

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._idle_since = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="motion-100hz")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── public API ────────────────────────────────────────────────

    def queue_move(self, move: Move) -> None:
        """Append a move to the primary queue.

        If breathing (or transitioning to breathing) is active, it is
        interrupted immediately so the new move starts ASAP.
        """
        with self._lock:
            if self._is_breathing:
                self._current_move = None
                self._queue.clear()
                self._is_breathing = False
            self._queue.append(move)
            self._sent_idle = False

    def clear_queue(self) -> None:
        """Stop all primary motion."""
        with self._lock:
            self._queue.clear()
            self._current_move = None
            self._is_breathing = False
            self._idle_since = time.monotonic()
            self._sent_idle = False

    def set_face_offsets(self, head_offset: npt.NDArray[np.float64]) -> None:
        with self._face_lock:
            self._face_offsets = head_offset.copy()
            self._face_offsets_dirty = True

    def clear_face_offsets(self) -> None:
        with self._face_lock:
            self._face_offsets = _IDENTITY_4x4.copy()
            self._face_offsets_dirty = True

    @property
    def current_pose(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Snapshot of the last evaluated primary pose (thread-safe)."""
        with self._lock:
            return (
                self._last_head.copy(),
                self._last_antennas.copy(),
                self._last_body_yaw,
            )

    @property
    def is_breathing(self) -> bool:
        with self._lock:
            return self._is_breathing

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return (
                self._current_move is not None
                and not isinstance(self._current_move, BreathingMove)
            )

    # ── 100 Hz loop ──────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            t_start = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t_start
            sleep_time = _TICK_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self) -> None:
        now = time.monotonic()

        has_move = False
        with self._lock:
            if self._current_move is not None:
                t = now - self._move_start
                if t >= self._current_move.duration:
                    self._current_move = None
                    self._idle_since = now
                    self._is_breathing = False

            if self._current_move is None and self._queue:
                self._current_move = self._queue.popleft()
                self._move_start = now

            if self._current_move is not None:
                has_move = True
                t = now - self._move_start
                t = min(t, self._current_move.duration - 1e-6)
                head, antennas, body_yaw = self._current_move.evaluate(t)
                if head is not None:
                    self._last_head = head
                if antennas is not None:
                    self._last_antennas = np.asarray(antennas, dtype=np.float64)
                if body_yaw is not None:
                    self._last_body_yaw = body_yaw

            snapshot_head = self._last_head.copy()
            snapshot_antennas = self._last_antennas.copy()
            snapshot_body_yaw = self._last_body_yaw

        with self._face_lock:
            face_dirty = self._face_offsets_dirty
            self._face_offsets_dirty = False
            face_offset = self._face_offsets.copy()

        need_send = has_move or face_dirty or not self._sent_idle
        if not need_send:
            return

        final_head = compose_world_offset(snapshot_head, face_offset)

        try:
            self._reachy.set_target(
                head=final_head,
                antennas=snapshot_antennas,
                body_yaw=snapshot_body_yaw,
            )
            if not has_move and not face_dirty:
                self._sent_idle = True
        except Exception:
            pass

    def _start_breathing(self, now: float) -> None:
        """Insert GotoMove→neutral then BreathingMove. Caller holds _lock."""
        goto = GotoMove(
            start_head_pose=self._last_head.copy(),
            target_head_pose=_NEUTRAL_HEAD.copy(),
            start_antennas=self._last_antennas.copy(),
            target_antennas=_NEUTRAL_ANTENNAS.copy(),
            start_body_yaw=self._last_body_yaw,
            target_body_yaw=_NEUTRAL_BODY_YAW,
            duration=_GOTO_NEUTRAL_DURATION,
            method=InterpolationTechnique.MIN_JERK,
        )
        self._current_move = goto
        self._move_start = now
        self._queue.append(BreathingMove())
        self._is_breathing = True
