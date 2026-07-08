from types import SimpleNamespace

from app import pipeline


def test_aec_loads_webrtc_for_selected_physical_sink(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:4] == ["pactl", "list", "short", "modules"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:3] == ["pactl", "load-module", "module-echo-cancel"]:
            return SimpleNamespace(returncode=0, stdout="42\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    aec = pipeline.PulseAudioAEC("reachy-physical-mic")

    assert aec.start("selected-physical-speaker")
    assert aec.active
    load_call = next(call for call in calls if call[:3] == ["pactl", "load-module", "module-echo-cancel"])
    assert "aec_method=webrtc" in load_call
    assert "source_master=reachy-physical-mic" in load_call
    assert "sink_master=selected-physical-speaker" in load_call
    assert f"source_name={pipeline.AEC_SOURCE_NAME}" in load_call
    assert f"sink_name={pipeline.AEC_SINK_NAME}" in load_call

    aec.stop()
    assert ["pactl", "unload-module", "42"] in calls


def test_virtual_aec_sink_is_hidden_from_speaker_selector(monkeypatch):
    stdout = (
        "0\tphysical-speaker\tmodule-alsa-card.c\ts16le 2ch 48000Hz\tIDLE\n"
        "1\treachy_aec_sink\tmodule-echo-cancel.c\tfloat32le 1ch 32000Hz\tRUNNING\n"
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    assert pipeline.list_pa_sinks() == [
        {"id": "physical-speaker", "label": "physical-speaker"},
    ]
