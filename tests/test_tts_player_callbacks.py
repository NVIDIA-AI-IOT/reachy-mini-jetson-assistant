import queue

import numpy as np

from app import pipeline


class FakeTTS:
    def synthesize(self, text):
        return {"audio": np.ones(10, dtype=np.int16), "sample_rate": 16000}


def test_tts_callbacks_wrap_actual_audio_once(monkeypatch):
    events = []
    monkeypatch.setattr(
        pipeline,
        "play_audio",
        lambda audio, sample_rate, sink=None: events.append(("play", sink)),
    )
    q = queue.Queue()
    q.put("first sentence")
    q.put("second sentence")
    q.put(None)

    pipeline.tts_player(
        FakeTTS(),
        q,
        sink="reachy-speaker",
        on_audio_start=lambda: events.append(("start", None)),
        on_audio_end=lambda: events.append(("end", None)),
    )

    assert events == [
        ("start", None),
        ("play", "reachy-speaker"),
        ("play", "reachy-speaker"),
        ("end", None),
    ]


def test_no_callbacks_when_tts_produces_no_audio(monkeypatch):
    class EmptyTTS:
        def synthesize(self, text):
            return {"audio": None, "sample_rate": 16000}

    events = []
    monkeypatch.setattr(pipeline, "play_audio", lambda *args, **kwargs: events.append("play"))
    q = queue.Queue()
    q.put("silence")
    q.put(None)

    pipeline.tts_player(
        EmptyTTS(), q,
        on_audio_start=lambda: events.append("start"),
        on_audio_end=lambda: events.append("end"),
    )

    assert events == []


def test_tts_resolves_live_speaker_for_each_audio_chunk(monkeypatch):
    played_sinks = []
    selected = {"sink": "external-speaker"}

    def fake_play(audio, sample_rate, sink=None):
        played_sinks.append(sink)
        selected["sink"] = "reachy-speaker"

    monkeypatch.setattr(pipeline, "play_audio", fake_play)
    q = queue.Queue()
    q.put("first sentence")
    q.put("second sentence")
    q.put(None)

    pipeline.tts_player(FakeTTS(), q, sink=lambda: selected["sink"])

    assert played_sinks == ["external-speaker", "reachy-speaker"]
