# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from app import pipeline


SPEAKERS = [
    {"id": "alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio", "label": "Reachy Mini"},
    {"id": "alsa_output.usb-Generic_Conference_Speaker", "label": "Generic Conference Speaker"},
]


def test_speaker_selector_prefers_configured_external_output(monkeypatch):
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: SPEAKERS)
    monkeypatch.setattr(
        pipeline,
        "get_default_pa_sink",
        lambda: SPEAKERS[0]["id"],
    )

    selector = pipeline.SpeakerSelector(
        preferred_hint="Generic Conference Speaker",
        fallback_hint="Reachy Mini Audio",
    )

    assert selector.get_sink() == SPEAKERS[1]["id"]


def test_speaker_selector_auto_selects_external_sink(monkeypatch):
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: SPEAKERS)
    monkeypatch.setattr(pipeline, "get_default_pa_sink", lambda: SPEAKERS[0]["id"])

    selector = pipeline.SpeakerSelector(None, "Reachy Mini Audio")

    assert selector.get_sink() == SPEAKERS[1]["id"]


def test_speaker_selector_switches_without_restart(monkeypatch):
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: SPEAKERS)
    monkeypatch.setattr(
        pipeline,
        "get_default_pa_sink",
        lambda: SPEAKERS[1]["id"],
    )
    pactl_calls = []

    def fake_run(args, **kwargs):
        pactl_calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    selector = pipeline.SpeakerSelector("Generic Conference Speaker", "Reachy Mini Audio")

    state = selector.select(SPEAKERS[0]["id"])

    assert state["selected"] == SPEAKERS[0]["id"]
    assert [
        "pactl", "set-default-sink", SPEAKERS[0]["id"],
    ] in pactl_calls


def test_speaker_selector_reports_and_sets_volume(monkeypatch):
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: SPEAKERS)
    monkeypatch.setattr(
        pipeline,
        "get_default_pa_sink",
        lambda: SPEAKERS[0]["id"],
    )
    pactl_calls = []

    def fake_run(args, **kwargs):
        pactl_calls.append(args)
        if args[:2] == ["pactl", "get-sink-volume"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Volume: front-left: 32768 / 50% / -18.06 dB, "
                    "front-right: 32768 / 50% / -18.06 dB\\n"
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    selector = pipeline.SpeakerSelector("Reachy Mini Audio", None)

    assert selector.state()["volume"] == 50
    state = selector.set_volume(55)

    assert state["volume"] == 50
    assert [
        "pactl", "set-sink-volume", SPEAKERS[0]["id"], "55%",
    ] in pactl_calls
