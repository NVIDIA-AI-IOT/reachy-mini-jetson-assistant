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

"""Movements — emotion-driven robot behaviors for Reachy Mini.

Uses the reachy-mini SDK's RecordedMoves library (81 pre-recorded moves
from pollen-robotics/reachy-mini-emotions-library) queued through the
MovementManager's 100 Hz control loop for seamless blending with face
tracking offsets.
"""

import random
import time
from typing import Optional

import numpy as np

from app.emotion import Emotion
from app.movement_manager import MovementManager

try:
    from reachy_mini.motion.goto import GotoMove
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini.utils.interpolation import InterpolationTechnique

    HAS_RECORDED_MOVES = True
except ImportError:
    HAS_RECORDED_MOVES = False

_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
_GOTO_TRANSITION_SECS = 0.5

EMOTION_MOVES: dict[Emotion, list[str]] = {
    Emotion.HAPPY: [
        "cheerful1", "laughing1", "laughing2",
        "relief1", "relief2", "enthusiastic1", "proud1",
    ],
    Emotion.SAD: ["sad1", "sad2", "downcast1", "lonely1", "resigned1"],
    Emotion.SURPRISED: ["surprised1", "surprised2", "amazed1"],
    Emotion.ANGRY: [
        "furious1", "rage1", "irritated1", "irritated2",
        "displeased1", "displeased2",
    ],
    Emotion.DISGUSTED: ["disgusted1", "contempt1", "go_away1"],
    Emotion.SCARED: ["fear1", "scared1", "anxiety1"],
    Emotion.CONTEMPT: ["contempt1", "indifferent1", "reprimand1", "boredom1"],
    Emotion.GREETING: ["welcoming1", "welcoming2", "come1"],
    Emotion.FAREWELL: ["calming1", "serenity1", "understanding1"],
}


class MovementController:
    """Emotion-driven movement controller for Reachy Mini.

    Queues GotoMove transitions and RecordedMoves into the
    MovementManager so movements blend smoothly with face tracking.
    No background threads — the MovementManager's 100 Hz loop
    handles execution.
    """

    MIN_CONFIDENCE = 0.75
    COOLDOWN_SECS = 8.0

    def __init__(self, manager: MovementManager, antenna_rest: list[float] | None = None):
        self._manager = manager
        self._antenna_rest = np.array(antenna_rest or [0.0, 0.0], dtype=np.float64)
        self._last_emotion: Optional[Emotion] = None
        self._last_react_time: float = 0.0
        self._moves: Optional["RecordedMoves"] = None

        if HAS_RECORDED_MOVES:
            try:
                self._moves = RecordedMoves(_EMOTIONS_DATASET)
            except Exception as e:
                print(f"  RecordedMoves load failed: {e}")

    def react(self, emotion: Emotion, confidence: float = 1.0) -> bool:
        """Queue a recorded move for the given emotion.

        Returns True if a movement was queued, False if suppressed.
        """
        if self._manager is None or self._moves is None:
            return False

        if emotion == Emotion.NEUTRAL:
            return False

        if confidence < self.MIN_CONFIDENCE:
            return False

        now = time.time()
        if (
            emotion == self._last_emotion
            and (now - self._last_react_time) < self.COOLDOWN_SECS
        ):
            return False

        move_names = EMOTION_MOVES.get(emotion)
        if not move_names:
            return False

        available = [n for n in move_names if n in self._moves.moves]
        if not available:
            return False

        self._last_emotion = emotion
        self._last_react_time = now
        move_name = random.choice(available)

        recorded = self._moves.get(move_name)

        start_head, start_ant, start_yaw = self._manager.current_pose
        target_head, target_ant, target_yaw = recorded.evaluate(0.0)

        goto = GotoMove(
            start_head_pose=start_head,
            target_head_pose=target_head,
            start_antennas=start_ant,
            target_antennas=np.asarray(target_ant, dtype=np.float64),
            start_body_yaw=start_yaw,
            target_body_yaw=float(target_yaw),
            duration=_GOTO_TRANSITION_SECS,
            method=InterpolationTechnique.MIN_JERK,
        )

        self._manager.queue_move(goto)
        self._manager.queue_move(recorded)
        return True

    def reset(self) -> None:
        """Clear all queued moves and return to neutral."""
        self._manager.clear_queue()

        start_head, start_ant, start_yaw = self._manager.current_pose
        goto = GotoMove(
            start_head_pose=start_head,
            target_head_pose=np.eye(4, dtype=np.float64),
            start_antennas=start_ant,
            target_antennas=self._antenna_rest.copy(),
            start_body_yaw=start_yaw,
            target_body_yaw=0.0,
            duration=0.4,
            method=InterpolationTechnique.MIN_JERK,
        )
        self._manager.queue_move(goto)

    @property
    def is_moving(self) -> bool:
        return self._manager.is_moving

    @property
    def last_emotion(self) -> Optional[Emotion]:
        return self._last_emotion
