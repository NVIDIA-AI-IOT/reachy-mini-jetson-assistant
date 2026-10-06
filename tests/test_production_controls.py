# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import stat

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.audit import AuditLogger
from app.config import Config
from app.web import Broadcaster, create_app


TOKEN = "a" * 48


def _client(broadcaster=None):
    broadcaster = broadcaster or Broadcaster()
    app = create_app(
        broadcaster,
        api_token=TOKEN,
        require_auth=True,
        allowed_hosts=["testserver"],
        max_clients=2,
    )
    return TestClient(app), broadcaster


def test_http_authentication_and_security_headers():
    client, _ = _client()
    with client:
        rejected = client.get("/")
        assert rejected.status_code == 401
        assert rejected.json() == {"detail": "unauthorized"}

        allowed = client.get(f"/?token={TOKEN}")
        assert allowed.status_code == 200
        assert allowed.headers["x-frame-options"] == "DENY"
        assert allowed.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in allowed.headers["content-security-policy"]
        assert "script-src 'self' 'sha256-" in allowed.headers["content-security-policy"]
        assert "script-src 'self' 'unsafe-inline'" not in allowed.headers[
            "content-security-policy"
        ]


def test_liveness_and_readiness_are_public_but_minimal():
    client, broadcaster = _client()
    with client:
        assert client.get("/health/live").json() == {"status": "alive"}
        not_ready = client.get("/health/ready")
        assert not_ready.status_code == 503

        broadcaster.set_ready(True, {"stt": True, "vlm": True, "tts": True})
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["components"] == {
            "stt": True,
            "vlm": True,
            "tts": True,
        }


def test_websocket_requires_token_and_accepts_bounded_ptt_messages():
    client, _ = _client()
    with client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass

        with client.websocket_connect(f"/ws?token={TOKEN}") as websocket:
            assert websocket.receive_json() == {"type": "ptt_state", "active": False}
            assert websocket.receive_json() == {
                "type": "speaker_state",
                "speakers": [],
                "selected": None,
            }
            websocket.send_json({"type": "ptt", "active": True})
            assert websocket.receive_json() == {"type": "ptt_state", "active": True}


def test_authenticated_volume_control_uses_auto_selected_external_speaker(monkeypatch):
    from types import SimpleNamespace

    from rich.console import Console

    from app import pipeline

    speakers = [
        {"id": "alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio", "label": "Reachy Mini"},
        {"id": "alsa_output.usb-External_Speaker", "label": "External Speaker"},
    ]
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: speakers)
    monkeypatch.setattr(pipeline, "get_default_pa_sink", lambda: speakers[0]["id"])
    volumes = {speaker["id"]: 50 for speaker in speakers}
    monkeypatch.setattr(pipeline, "get_pa_sink_volume", lambda sink: volumes[sink])
    calls = []

    def set_volume(args, **kwargs):
        calls.append(args)
        volumes[args[2]] = int(args[3].rstrip("%"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", set_volume)
    mic = pipeline.MicRecorder(Console(), chunk_ms=32)
    mic.speaker_selector = pipeline.SpeakerSelector(None, "Reachy Mini Audio")
    client, broadcaster = _client()
    broadcaster.configure_speakers(
        mic.speaker_state, mic.select_speaker, mic.set_speaker_volume,
    )
    broadcaster.set_ready(True, {"microphone": True})

    with client:
        with client.websocket_connect(f"/ws?token={TOKEN}") as websocket:
            websocket.receive_json()  # Initial push-to-talk state.
            initial = websocket.receive_json()
            assert initial["selected"] == speakers[1]["id"]
            assert initial["volume"] == 50

            websocket.send_json({"type": "set_speaker_volume", "volume": 65})
            updated = websocket.receive_json()
            assert updated == {**initial, "volume": 65}
            assert calls == [["pactl", "set-sink-volume", speakers[1]["id"], "65%"]]
            assert volumes[speakers[0]["id"]] == 50
            assert client.get("/health/ready").status_code == 200

            websocket.send_json({"type": "ptt", "active": True})
            assert websocket.receive_json() == {"type": "ptt_state", "active": True}


def test_websocket_allows_lan_origin_when_all_hosts_are_allowed():
    broadcaster = Broadcaster()
    app = create_app(
        broadcaster,
        require_auth=False,
        allowed_hosts=["*"],
        max_clients=2,
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws", headers={"origin": "http://10.111.65.10:8090"}
        ) as websocket:
            assert websocket.receive_json() == {"type": "ptt_state", "active": False}


def test_production_configuration_fails_closed_without_secret():
    config = Config()
    config.runtime.production_mode = True
    config.llm.model = "pinned-model"
    config.web.host = "127.0.0.1"
    config.web.allowed_hosts = ["localhost"]
    config.web.require_auth = True
    config.web.api_token = ""

    errors = config.validation_errors()
    assert any("REACHY_WEB_API_TOKEN" in error for error in errors)

    config.web.api_token = TOKEN
    assert config.validation_errors() == []


def test_audit_log_rotates_privately_and_drops_sensitive_fields(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path), max_bytes=1_048_576, backup_count=2)
    audit.write(
        "guardrail",
        blocked=True,
        reasons=["prompt_leakage"],
        raw_response="must-not-be-written",
        api_token="must-not-be-written",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "guardrail"
    assert record["blocked"] is True
    assert record["reasons"] == ["prompt_leakage"]
    assert "raw_response" not in record
    assert "api_token" not in record
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
