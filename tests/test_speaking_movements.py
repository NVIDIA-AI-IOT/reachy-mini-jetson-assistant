import random

from app.speaking_movements import SpeakingMovementController


class FakeMove:
    duration = 1.0


class FakeLibrary:
    def __init__(self, names):
        self.moves = {name: FakeMove() for name in names}

    def get(self, name):
        return self.moves[name]


class FakeManager:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play_gesture(self, move):
        self.played.append(move)
        return True

    def stop_gesture(self):
        self.stopped += 1


def test_starts_one_available_gesture_and_stops_with_response():
    manager = FakeManager()
    library = FakeLibrary(["enthusiastic1", "welcoming1"])
    controller = SpeakingMovementController(
        manager,
        excitement_probability=1.0,
        library=library,
        rng=random.Random(3),
    )

    assert controller.available
    assert controller.start_response() == "enthusiastic1"
    assert manager.played == [library.get("enthusiastic1")]
    controller.stop_response()
    assert manager.stopped == 1
    assert controller.active_move is None


def test_does_not_immediately_repeat_when_pool_has_alternatives():
    manager = FakeManager()
    library = FakeLibrary(["welcoming1", "welcoming2"])
    controller = SpeakingMovementController(
        manager,
        excitement_probability=0.0,
        library=library,
        rng=random.Random(1),
    )

    first = controller.start_response()
    controller.stop_response()
    second = controller.start_response()

    assert first in library.moves
    assert second in library.moves
    assert second != first


def test_unavailable_when_no_curated_recording_exists():
    manager = FakeManager()
    controller = SpeakingMovementController(manager, library=FakeLibrary(["sad1"]))

    assert not controller.available
    assert controller.start_response() is None
    assert manager.played == []
