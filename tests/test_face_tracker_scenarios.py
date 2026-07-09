import time
import unittest

from app.face_tracker import FaceTracker


FRAME_SHAPE = (1080, 1920, 3)
CENTER_FACE = (840, 390, 1080, 690)
MODERATE_LEFT_FACE = (552, 390, 792, 690)
FAR_LEFT_FACE = (0, 390, 240, 690)
FAR_RIGHT_FACE = (1680, 390, 1920, 690)
MID_RIGHT_FACE = (1224, 390, 1464, 690)
FAR_TOP_FACE = (840, 0, 1080, 300)
FAR_BOTTOM_FACE = (840, 780, 1080, 1080)


class FakeMovementManager:
    def __init__(self):
        self.targets = []
        self.holds = 0

    def set_targets(self, body, pitch, yaw):
        self.targets.append((body, pitch, yaw))

    def hold_current(self):
        self.holds += 1

    def reset(self):
        pass


def make_tracker(**overrides):
    manager = FakeMovementManager()
    options = {
        "fps": 15.0,
        "dead_zone": 0.12,
        "lock_zone": 0.18,
        "reacquire_zone": 0.45,
        "good_frame_zone": 0.18,
        "head_yaw_max_deg": 20.0,
        "head_yaw_gain": 18.0,
        "head_yaw_step": 1.4,
        "soft_center_head_yaw_max_deg": 12.0,
        "soft_center_head_yaw_step": 0.75,
        "body_max_deg": 30.0,
        "body_gain": 18.0,
        "body_step": 0.7,
        "invert_body": True,
        "body_enabled": True,
        "vertical": False,
        "return_to_neutral": False,
        "scan_enabled": True,
        "scan_body_range_deg": 20.0,
        "scan_speed_deg_per_sec": 5.0,
    }
    options.update(overrides)
    return FaceTracker(None, None, manager, **options), manager


class FaceTrackerScenarioTests(unittest.TestCase):
    def test_no_face_starts_a_slow_bounded_search(self):
        tracker, manager = make_tracker()

        tracker._handle_face_lost(time.monotonic())

        self.assertTrue(tracker.is_scanning)
        body, pitch, yaw = manager.targets[-1]
        self.assertAlmostEqual(body, 5.0 * 0.35 / 15.0)
        self.assertEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 5.0 / 15.0)

    def test_search_sweep_never_exceeds_head_or_body_limits(self):
        tracker, manager = make_tracker()

        for _ in range(500):
            tracker._scan_for_face()

        bodies = [body for body, _, _ in manager.targets]
        yaws = [yaw for _, _, yaw in manager.targets]
        self.assertLessEqual(max(map(abs, bodies)), 20.0)
        self.assertLessEqual(max(map(abs, yaws)), 20.0)
        self.assertLess(min(bodies), 0.0)
        self.assertGreater(max(bodies), 0.0)
        self.assertLess(min(yaws), 0.0)
        self.assertGreater(max(yaws), 0.0)

    def test_brief_detection_dropout_holds_before_searching(self):
        tracker, manager = make_tracker(face_lost_delay=3.0)
        now = time.monotonic()
        tracker._last_face_time = now

        tracker._handle_face_lost(now + 2.9)
        self.assertEqual(manager.targets, [])

        tracker._handle_face_lost(now + 3.1)
        self.assertTrue(tracker.is_scanning)
        self.assertEqual(len(manager.targets), 1)

    def test_far_left_face_uses_head_and_slow_body_assist(self):
        tracker, manager = make_tracker()

        tracker._servo(FAR_LEFT_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (0.7, 0.0, 1.4))
        self.assertEqual(tracker.target_yaw_deg, 1.4)
        self.assertEqual(tracker.target_body_yaw_deg, 0.7)

    def test_centered_face_moving_left_uses_gentle_head_only_first(self):
        tracker, manager = make_tracker()
        tracker._servo(CENTER_FACE, FRAME_SHAPE)

        tracker._servo(MODERATE_LEFT_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (0.0, 0.0, 0.75))

    def test_far_edge_commands_remain_bounded(self):
        tracker, manager = make_tracker()

        for _ in range(100):
            tracker._servo(FAR_LEFT_FACE, FRAME_SHAPE)

        self.assertEqual(tracker.target_yaw_deg, 20.0)
        self.assertEqual(tracker.target_body_yaw_deg, 30.0)
        self.assertLessEqual(abs(tracker.target_body_yaw_deg), 30.0)
        self.assertTrue(all(abs(body) <= 30.0 for body, _, _ in manager.targets))
        self.assertTrue(all(abs(yaw) <= 20.0 for _, _, yaw in manager.targets))
        self.assertTrue(all(pitch == 0.0 for _, pitch, _ in manager.targets))

    def test_far_right_face_accumulates_to_opposite_limits(self):
        tracker, _ = make_tracker()

        for _ in range(100):
            tracker._servo(FAR_RIGHT_FACE, FRAME_SHAPE)

        self.assertEqual(tracker.target_yaw_deg, -20.0)
        self.assertEqual(tracker.target_body_yaw_deg, -30.0)

    def test_face_above_center_tilts_head_up_without_rotating_body(self):
        tracker, manager = make_tracker(vertical=True)

        tracker._servo(FAR_TOP_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (0.0, -1.2, 0.0))
        self.assertTrue(tracker._reacquiring_y)
        self.assertFalse(tracker._reacquiring_x)

    def test_face_below_center_tilts_head_down_without_rotating_body(self):
        tracker, manager = make_tracker(vertical=True)

        tracker._servo(FAR_BOTTOM_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (0.0, 1.2, 0.0))
        self.assertTrue(tracker._reacquiring_y)
        self.assertFalse(tracker._reacquiring_x)

    def test_vertical_pitch_accumulates_but_remains_bounded(self):
        tracker, manager = make_tracker(vertical=True)

        for _ in range(100):
            tracker._servo(FAR_TOP_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (0.0, -18.0, 0.0))
        self.assertTrue(all(body == 0.0 for body, _, _ in manager.targets))
        self.assertTrue(all(yaw == 0.0 for _, _, yaw in manager.targets))
        self.assertTrue(all(abs(pitch) <= 18.0 for _, pitch, _ in manager.targets))

    def test_horizontal_search_preserves_last_vertical_angle(self):
        tracker, manager = make_tracker(vertical=True)
        tracker._pitch = -7.0

        tracker._scan_for_face()

        self.assertEqual(manager.targets[-1][1], -7.0)

    def test_reacquisition_continues_below_entry_threshold_until_centered(self):
        tracker, manager = make_tracker()
        tracker._servo(FAR_RIGHT_FACE, FRAME_SHAPE)
        self.assertTrue(tracker._reacquiring)

        # Error is now +0.40: below the +0.45 reacquire entry threshold but
        # still outside the +0.18 good-frame zone. Wide tracking must remain.
        tracker._servo(MID_RIGHT_FACE, FRAME_SHAPE)

        self.assertTrue(tracker._reacquiring)
        self.assertEqual(manager.targets[-1], (-1.4, 0.0, -2.8))

        tracker._servo(CENTER_FACE, FRAME_SHAPE)
        self.assertFalse(tracker._reacquiring)

    def test_centered_face_rebalances_head_yaw_into_body(self):
        tracker, manager = make_tracker()
        tracker._body = -13.0
        tracker._yaw = -20.0

        tracker._servo(CENTER_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets[-1], (-13.7, 0.0, -19.3))
        self.assertEqual(tracker.target_body_yaw_deg + tracker.target_yaw_deg, -33.0)

        for _ in range(39):
            tracker._servo(CENTER_FACE, FRAME_SHAPE)

        self.assertAlmostEqual(tracker.target_body_yaw_deg, -30.0)
        self.assertAlmostEqual(tracker.target_yaw_deg, -3.0)
        self.assertAlmostEqual(tracker.target_body_yaw_deg + tracker.target_yaw_deg, -33.0)

    def test_rebalance_holds_when_body_is_at_limit(self):
        tracker, manager = make_tracker()
        tracker._body = -30.0
        tracker._yaw = -3.0

        tracker._servo(CENTER_FACE, FRAME_SHAPE)

        self.assertEqual(manager.targets, [])
        self.assertEqual(manager.holds, 1)


if __name__ == "__main__":
    unittest.main()
