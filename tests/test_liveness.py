from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liveness import (  # noqa: E402
    SignalFusion,
    aggregate_track_scores,
    analyze_screen_presentation,
    fuse_signal_scores,
    same_face_track,
)
from vision import detect_faces_yunet  # noqa: E402


class TrackAggregationTests(unittest.TestCase):
    def test_uses_lower_quartile_not_median_only(self) -> None:
        scores = deque([0.95, 0.92, 0.90, 0.88, 0.20])

        aggregate = aggregate_track_scores(scores)

        self.assertLess(aggregate, 0.80)
        self.assertGreater(aggregate, 0.50)

    def test_track_rejects_large_identity_jump(self) -> None:
        self.assertTrue(same_face_track((100, 100, 120, 120), (108, 102, 118, 121)))
        self.assertFalse(same_face_track((100, 100, 120, 120), (300, 100, 120, 120)))


class SignalFusionTests(unittest.TestCase):
    def test_accepts_only_when_every_required_signal_is_observed(self) -> None:
        result = fuse_signal_scores(
            model_score=0.88,
            weakest_model_score=0.82,
            screen_risk=0.10,
            motion_score=0.42,
            planar_risk=0.18,
            motion_observations=5,
            threshold=0.50,
            screen_risk_threshold=0.62,
            motion_min_observations=4,
        )

        self.assertIsInstance(result, SignalFusion)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reasons, ())

    def test_fails_closed_when_motion_is_unobserved(self) -> None:
        result = fuse_signal_scores(
            model_score=0.95,
            weakest_model_score=0.92,
            screen_risk=0.05,
            motion_score=0.0,
            planar_risk=0.0,
            motion_observations=0,
            threshold=0.50,
            screen_risk_threshold=0.62,
            motion_min_observations=4,
        )

        self.assertFalse(result.accepted)
        self.assertIn("motion not observed", result.reasons)

    def test_screen_risk_is_a_hard_rejection(self) -> None:
        result = fuse_signal_scores(
            model_score=0.99,
            weakest_model_score=0.99,
            screen_risk=0.90,
            motion_score=0.80,
            planar_risk=0.0,
            motion_observations=8,
            threshold=0.50,
            screen_risk_threshold=0.62,
            motion_min_observations=4,
        )

        self.assertFalse(result.accepted)
        self.assertIn("screen cues", result.reasons)


class ScreenCueTests(unittest.TestCase):
    def test_full_frame_rectangle_containing_face_is_detected(self) -> None:
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        cv2.rectangle(frame, (145, 55), (495, 425), (245, 245, 245), 8)
        cv2.rectangle(frame, (155, 65), (485, 415), (100, 130, 170), -1)
        face = (245, 145, 120, 140)

        signals = analyze_screen_presentation(frame, face)

        self.assertGreaterEqual(signals.rectangle, 0.8)
        self.assertGreaterEqual(signals.risk, 0.62)
        self.assertIn("screenMoire", signals.metadata)


class _FakeYuNet:
    def setInputSize(self, _size: tuple[int, int]) -> None:
        pass

    def detect(self, _frame: np.ndarray) -> tuple[None, np.ndarray]:
        return (
            None,
            np.asarray(
                [
                    [
                        100,
                        80,
                        120,
                        130,
                        130,
                        120,
                        185,
                        120,
                        158,
                        150,
                        138,
                        180,
                        180,
                        180,
                        0.99,
                    ],
                ],
                dtype=np.float32,
            ),
        )


class FaceLandmarkTests(unittest.TestCase):
    def test_yunet_landmarks_are_kept_for_temporal_analysis(self) -> None:
        detections = detect_faces_yunet(
            _FakeYuNet(),
            np.zeros((300, 400, 3), dtype=np.uint8),
            90,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(len(detections[0].landmarks), 5)
        self.assertEqual(detections[0].landmarks[2], (158.0, 150.0))


if __name__ == "__main__":
    unittest.main()
