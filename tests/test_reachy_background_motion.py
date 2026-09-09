# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from app.reachy import disable_sdk_background_motion


class ModernSDKReachy:
    def __init__(self):
        self.calls = []

    def disable_wobbling(self):
        self.calls.append("disable_wobbling")

    def stop_head_tracking(self):
        self.calls.append("stop_head_tracking")


def test_disables_modern_sdk_background_motion_features():
    reachy = ModernSDKReachy()

    disabled = disable_sdk_background_motion(reachy)

    assert disabled == ["disable_wobbling", "stop_head_tracking"]
    assert reachy.calls == disabled


def test_legacy_sdk_without_background_motion_features_is_supported():
    class LegacySDKReachy:
        pass

    assert disable_sdk_background_motion(LegacySDKReachy()) == []
