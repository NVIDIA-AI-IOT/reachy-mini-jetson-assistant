# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from app.movement_manager import MovementManager


class FakeReachy:
    def __init__(self):
        self.calls = []

    def set_target(self, **kwargs):
        self.calls.append(kwargs)


def test_stationary_target_uses_configured_antenna_rest_bias():
    reachy = FakeReachy()
    manager = MovementManager(
        reachy,
        antenna_rest_position=[-0.1745, 0.1745],
    )

    manager._tick()

    np.testing.assert_allclose(
        reachy.calls[-1]["antennas"], [-0.1745, 0.1745]
    )


def test_antenna_rest_bias_requires_two_joint_angles():
    with pytest.raises(ValueError, match="exactly two"):
        MovementManager(FakeReachy(), antenna_rest_position=[0.0])
