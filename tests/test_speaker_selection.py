from types import SimpleNamespace

from app import pipeline


SPEAKERS = [
    {"id": "alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio", "label": "Reachy Mini"},
    {"id": "alsa_output.usb-Anker_PowerConf", "label": "Anker PowerConf"},
]


def test_speaker_selector_prefers_configured_external_output(monkeypatch):
    monkeypatch.setattr(pipeline, "list_pa_sinks", lambda: SPEAKERS)
    monkeypatch.setattr(
        pipeline,
        "get_default_pa_sink",
        lambda: SPEAKERS[0]["id"],
    )

    selector = pipeline.SpeakerSelector(
        preferred_hint="Anker PowerConf",
        fallback_hint="Reachy Mini Audio",
    )

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
    selector = pipeline.SpeakerSelector("Anker PowerConf", "Reachy Mini Audio")

    state = selector.select(SPEAKERS[0]["id"])

    assert state["selected"] == SPEAKERS[0]["id"]
    assert pactl_calls[-1] == ["pactl", "set-default-sink", SPEAKERS[0]["id"]]
