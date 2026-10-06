# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.config import Config


def _clear_platform_environment(monkeypatch):
    monkeypatch.delenv("REACHY_PLATFORM", raising=False)
    monkeypatch.delenv("REACHY_ASSISTANT_CONFIG", raising=False)
    monkeypatch.delenv("REACHY_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("REACHY_WEB_API_TOKEN", raising=False)


def test_orin_nano_keeps_upstream_jetpack_72_defaults(monkeypatch):
    _clear_platform_environment(monkeypatch)
    config = Config.load()

    assert config.platform.profile == "orin_nano"
    assert config.platform.supported is True
    assert config.platform.model_server == "llama_cpp"
    assert config.llm.base_url == "http://localhost:8080"
    assert config.llm.model == ""
    assert config.platform.jetpack == "7.2"
    assert config.stt.compute_type == "float16"
    assert config.audio.output_device == "Reachy Mini Audio"
    assert config.audio.echo_cancellation is True
    assert config.vision.camera_device == 1
    assert config.vision.history_turns == 5
    assert config.runtime.production_mode is False
    assert config.web.require_auth is False


def test_thor_profile_applies_validated_overrides(monkeypatch):
    _clear_platform_environment(monkeypatch)
    config = Config.load(platform="thor")

    assert config.platform.display_name == "Jetson AGX Thor"
    assert config.platform.model_server == "vllm"
    assert config.llm.model == "google/gemma-4-E4B-it"
    assert config.llm.base_url == "http://localhost:8001"
    assert config.stt.model == "large-v3"
    assert config.stt.compute_type == "float16"
    assert config.stt.beam_size == 5
    assert config.audio.output_device is None
    assert config.audio.echo_cancellation is False
    assert config.vision.camera_device == -1
    assert config.vision.history_turns == 5
    assert config.runtime.production_mode is False
    assert config.web.host == "0.0.0.0"
    assert config.web.require_auth is False
    assert config.web.max_clients == 8
    assert config.validation_errors() == []


def test_agx_orin_profile_is_architected_but_fails_closed(monkeypatch):
    _clear_platform_environment(monkeypatch)
    config = Config.load(platform="agx_orin")

    assert config.platform.display_name == "Jetson AGX Orin"
    assert config.platform.implementation_status == "planned"
    assert config.platform.supported is False
    assert any("reserved but not implemented" in item for item in config.validation_errors())


def test_environment_and_aliases_select_profiles(monkeypatch):
    _clear_platform_environment(monkeypatch)
    monkeypatch.setenv("REACHY_PLATFORM", "orin-nano")
    assert Config.load().platform.profile == "orin_nano"

    monkeypatch.setenv("REACHY_PLATFORM", "jetson-thor")
    assert Config.load().platform.profile == "thor"


def test_custom_config_overlays_selected_platform(tmp_path, monkeypatch):
    _clear_platform_environment(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "platform:\n  profile: thor\nllm:\n  temperature: 0.05\n",
        encoding="utf-8",
    )

    config = Config.load(str(custom))
    assert config.platform.profile == "thor"
    assert config.llm.base_url == "http://localhost:8001"
    assert config.llm.temperature == 0.05


def test_unknown_platform_is_rejected(monkeypatch):
    _clear_platform_environment(monkeypatch)
    with pytest.raises(ValueError, match="Unknown Jetson platform profile"):
        Config.load(platform="future_device")


def test_thor_profile_combines_with_upstream_no_reachy_overlay(monkeypatch):
    from app.config import CONFIG_ROOT

    _clear_platform_environment(monkeypatch)
    monkeypatch.setenv("REACHY_PLATFORM", "thor")
    monkeypatch.setenv(
        "REACHY_ASSISTANT_CONFIG", str(CONFIG_ROOT / "settings.jp72-no-reachy.yaml")
    )

    config = Config.load()

    assert config.platform.profile == "thor"
    assert config.platform.jetpack == "7.1"
    assert config.llm.model == "google/gemma-4-E4B-it"
    assert config.llm.base_url == "http://localhost:8001"
    assert config.stt.model == "large-v3"
    assert config.stt.beam_size == 5
    assert config.stt.compute_type == "float16"
    assert config.reachy.enabled is False
    assert config.reachy.spawn_daemon is False
    assert config.audio.input_device is None
    assert config.rag.enabled is False
    assert config.validation_errors() == []


def test_explicit_config_overrides_environment_overlay(tmp_path, monkeypatch):
    from app.config import CONFIG_ROOT

    _clear_platform_environment(monkeypatch)
    monkeypatch.setenv("REACHY_PLATFORM", "thor")
    monkeypatch.setenv("REACHY_ASSISTANT_CONFIG", str(CONFIG_ROOT / "settings.jp6.yaml"))
    custom = tmp_path / "custom.yaml"
    custom.write_text("llm:\n  temperature: 0.05\n", encoding="utf-8")

    config = Config.load(str(custom))

    assert config.platform.profile == "thor"
    assert config.stt.compute_type == "float16"
    assert config.llm.temperature == 0.05


def test_legacy_overlay_preserves_jetpack_6_defaults(monkeypatch):
    from app.config import CONFIG_ROOT

    _clear_platform_environment(monkeypatch)
    config = Config.load(str(CONFIG_ROOT / "settings.jp6.yaml"), platform="orin_nano")

    assert config.platform.jetpack == "6.x"
    assert config.stt.compute_type == "int8"
    assert config.audio.output_device == "Anker PowerConf"
    assert config.audio.echo_cancellation is True
