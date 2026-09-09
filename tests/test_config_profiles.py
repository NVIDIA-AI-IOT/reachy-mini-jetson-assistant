# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from app.config import Config


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def test_default_config_targets_jetpack_72(monkeypatch):
    monkeypatch.delenv("REACHY_ASSISTANT_CONFIG", raising=False)

    config = Config.load()

    assert config.stt.device == "cuda"
    assert config.stt.compute_type == "float16"
    assert config.audio.input_device == "Reachy Mini Audio"
    assert config.audio.output_device == "Reachy Mini Audio"
    assert config.reachy.enabled is True
    assert config.reachy.antenna_rest_position == [-0.1745, 0.1745]


def test_jetpack_6_overlay_restores_legacy_runtime():
    config = Config.load(str(CONFIG_DIR / "settings.jp6.yaml"))

    assert config.stt.device == "cuda"
    assert config.stt.compute_type == "int8"
    assert config.audio.output_device == "Anker PowerConf"
    assert config.reachy.enabled is True


def test_no_reachy_overlay_disables_hardware_and_rag():
    config = Config.load(str(CONFIG_DIR / "settings.jp72-no-reachy.yaml"))

    assert config.stt.compute_type == "float16"
    assert config.audio.input_device is None
    assert config.audio.output_device is None
    assert config.audio.echo_cancellation is False
    assert config.reachy.enabled is False
    assert config.reachy.spawn_daemon is False
    assert config.rag.enabled is False


def test_conservative_reachy_overlay_limits_motion():
    config = Config.load(str(CONFIG_DIR / "settings.jp72-reachy.yaml"))

    assert config.stt.compute_type == "float16"
    assert config.reachy.enabled is True
    assert config.reachy.tracking_motion_enabled is False
    assert config.reachy.tracking_body_enabled is False
    assert config.reachy.tracking_scan_enabled is False
    assert config.reachy.speaking_movements_enabled is False


def test_environment_selects_overlay(monkeypatch):
    monkeypatch.setenv(
        "REACHY_ASSISTANT_CONFIG",
        str(CONFIG_DIR / "settings.jp6.yaml"),
    )

    config = Config.load()

    assert config.stt.compute_type == "int8"
