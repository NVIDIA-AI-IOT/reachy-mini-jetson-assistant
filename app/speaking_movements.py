# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curated official Pollen gestures that make spoken replies feel alive."""

import random
import threading
from typing import Optional

from app.movement_manager import MovementManager

try:
    from reachy_mini.motion.recorded_move import RecordedMoves

    HAS_RECORDED_MOVES = True
except ImportError:
    RecordedMoves = None
    HAS_RECORDED_MOVES = False


_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# Short, positive recordings only. Long dances, negative emotions, sleep,
# anger, fear, and dismissive gestures are intentionally excluded.
EXCITEMENT_MOVES = (
    "enthusiastic1",
    "enthusiastic2",
    "cheerful1",
    "success1",
    "success2",
    "amazed1",
)

INTERACTIVE_MOVES = (
    "welcoming1",
    "welcoming2",
    "attentive1",
    "attentive2",
    "curious1",
    "helpful1",
    "helpful2",
    "understanding1",
    "understanding2",
    "inquiring1",
    "inquiring2",
    "inquiring3",
    "yes1",
    "grateful1",
)


class SpeakingMovementController:
    """Choose one non-repeating official gesture for each spoken response."""

    def __init__(
        self,
        manager: MovementManager,
        *,
        excitement_probability: float = 0.4,
        library=None,
        rng: Optional[random.Random] = None,
    ):
        self._manager = manager
        self._excitement_probability = max(0.0, min(1.0, excitement_probability))
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._last_move: Optional[str] = None
        self._active_move: Optional[str] = None
        self._library = library

        if self._library is None and HAS_RECORDED_MOVES:
            try:
                self._library = RecordedMoves(_DATASET)
            except Exception as exc:
                print(f"Speaking movements unavailable: {exc}")

        available = self._library.moves if self._library is not None else {}
        self._excitement = tuple(name for name in EXCITEMENT_MOVES if name in available)
        self._interactive = tuple(name for name in INTERACTIVE_MOVES if name in available)

    @property
    def available(self) -> bool:
        return bool(self._excitement or self._interactive)

    @property
    def active_move(self) -> Optional[str]:
        with self._lock:
            return self._active_move

    def start_response(self) -> Optional[str]:
        """Start one gesture when the first TTS audio is about to play."""
        if not self.available:
            return None

        with self._lock:
            prefer_excitement = self._rng.random() < self._excitement_probability
            primary = self._excitement if prefer_excitement else self._interactive
            fallback = self._interactive if prefer_excitement else self._excitement
            candidates = primary or fallback
            without_repeat = tuple(name for name in candidates if name != self._last_move)
            move_name = self._rng.choice(without_repeat or candidates)
            recorded = self._library.get(move_name)
            if not self._manager.play_gesture(recorded):
                return None
            self._last_move = move_name
            self._active_move = move_name

        print(f"Speaking movement: {move_name}")
        return move_name

    def stop_response(self) -> None:
        """Return smoothly to tracking when the complete TTS reply ends."""
        with self._lock:
            was_active = self._active_move is not None
            self._active_move = None
        if was_active:
            self._manager.stop_gesture()
