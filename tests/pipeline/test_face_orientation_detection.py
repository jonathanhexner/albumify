"""Unit tests for face orientation detection.

Tests that the detect_face_orientation function correctly identifies
face orientation (0°, 90°, 180°, 270°) from 5-point landmarks.

These tests cover SIGHTING-001: Face Alignment Not Working.
"""

import pytest
import numpy as np

from sim_bench.pipeline.steps.detect_face_orientation import (
    detect_face_orientation,
    compute_orientation_confidence,
)


class TestDetectFaceOrientation:
    """Tests for detect_face_orientation function."""

    def test_upright_face_returns_0(self):
        """Upright face (eyes above nose above mouth) should return 0°."""
        # Standard upright face layout:
        # - Eyes at y=50
        # - Nose at y=100
        # - Mouth at y=150
        landmarks = [
            [100, 50],   # left_eye
            [200, 50],   # right_eye
            [150, 100],  # nose
            [110, 150],  # left_mouth
            [190, 150],  # right_mouth
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 0, f"Expected 0° for upright face, got {orientation}°"

    def test_upside_down_face_returns_180(self):
        """Upside-down face (eyes below nose below mouth) should return 180°.

        This is the critical test case from SIGHTING-001 (Face #118).
        """
        # Upside-down face layout:
        # - Eyes at y=150 (at BOTTOM)
        # - Nose at y=100 (ABOVE eyes - wrong!)
        # - Mouth at y=50 (at TOP - wrong!)
        landmarks = [
            [100, 150],  # left_eye (at BOTTOM)
            [200, 150],  # right_eye (at BOTTOM)
            [150, 100],  # nose (ABOVE eyes)
            [110, 50],   # left_mouth (at TOP)
            [190, 50],   # right_mouth (at TOP)
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 180, f"Expected 180° for upside-down face, got {orientation}°"

    def test_90_degree_cw_rotation_returns_270(self):
        """Face rotated 90° clockwise (top of head pointing right, mouth on left) should return 270°."""
        # When face is tilted 90° CW (top of head pointing RIGHT):
        # - Eyes are stacked vertically
        # - Mouth is on the LEFT of eyes (smaller x)
        # - Need to rotate 270° CW (= 90° CCW) to fix
        landmarks = [
            [150, 100],  # left_eye (on right side)
            [150, 200],  # right_eye (on right side)
            [100, 150],  # nose (center)
            [50, 110],   # left_mouth (on left side)
            [50, 190],   # right_mouth (on left side)
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 270, f"Expected 270° for CW rotated face, got {orientation}°"

    def test_90_degree_ccw_rotation_returns_90(self):
        """Face rotated 90° counter-clockwise (top of head pointing left, mouth on right) should return 90°."""
        # When face is tilted 90° CCW (top of head pointing LEFT):
        # - Eyes are stacked vertically
        # - Mouth is on the RIGHT of eyes (larger x)
        # - Need to rotate 90° CW to fix
        landmarks = [
            [50, 100],   # left_eye (on left side)
            [50, 200],   # right_eye (on left side)
            [100, 150],  # nose (center)
            [150, 110],  # left_mouth (on right side)
            [150, 190],  # right_mouth (on right side)
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 90, f"Expected 90° for CCW rotated face, got {orientation}°"

    def test_slightly_tilted_upright_face(self):
        """Slightly tilted but still upright face should return 0°."""
        # Eyes tilted ~15° but still clearly upright (eyes above nose above mouth)
        landmarks = [
            [90, 55],    # left_eye (slightly lower)
            [210, 45],   # right_eye (slightly higher)
            [150, 100],  # nose
            [110, 150],  # left_mouth
            [190, 150],  # right_mouth
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 0, f"Expected 0° for tilted upright face, got {orientation}°"

    def test_slightly_tilted_upside_down_face(self):
        """Slightly tilted but upside-down face should return 180°."""
        landmarks = [
            [90, 145],   # left_eye (at bottom, tilted)
            [210, 155],  # right_eye (at bottom, tilted)
            [150, 100],  # nose (above eyes)
            [110, 50],   # left_mouth (at top)
            [190, 50],   # right_mouth (at top)
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 180, f"Expected 180° for tilted upside-down face, got {orientation}°"

    def test_empty_landmarks_returns_0(self):
        """Empty landmarks should return 0° (no rotation)."""
        assert detect_face_orientation([]) == 0
        assert detect_face_orientation(None) == 0

    def test_insufficient_landmarks_returns_0(self):
        """Less than 5 landmarks should return 0°."""
        landmarks = [[100, 50], [200, 50], [150, 100]]  # Only 3 points
        assert detect_face_orientation(landmarks) == 0

    def test_face_118_regression(self):
        """Regression test for Face #118 from SIGHTING-001.

        This face was identified with eyes at bottom, mouth/nose at top,
        but was only rotated 6° instead of ~180°.
        """
        # Simulated landmarks for Face #118 (upside-down face)
        # Eyes are at bottom (high y), mouth at top (low y)
        landmarks = [
            [245, 380],  # left_eye (at bottom of face region)
            [295, 378],  # right_eye (at bottom)
            [268, 330],  # nose (above eyes)
            [248, 285],  # left_mouth (at top)
            [288, 283],  # right_mouth (at top)
        ]
        orientation = detect_face_orientation(landmarks)
        assert orientation == 180, (
            f"Face #118 regression: Expected 180° for upside-down face, got {orientation}°. "
            "This was the root cause of SIGHTING-001."
        )


class TestOrientationConfidence:
    """Tests for compute_orientation_confidence function."""

    def test_perfect_upright_has_high_confidence(self):
        """Perfect upright face should have confidence 1.0."""
        landmarks = [
            [100, 50],
            [200, 50],
            [150, 100],
            [110, 150],
            [190, 150],
        ]
        confidence = compute_orientation_confidence(landmarks, 0)
        assert confidence == 1.0

    def test_perfect_upside_down_has_high_confidence(self):
        """Perfect upside-down face should have confidence 1.0."""
        landmarks = [
            [100, 150],
            [200, 150],
            [150, 100],
            [110, 50],
            [190, 50],
        ]
        confidence = compute_orientation_confidence(landmarks, 180)
        assert confidence == 1.0

    def test_wrong_orientation_has_low_confidence(self):
        """Wrong orientation should have low confidence."""
        # Upright landmarks
        landmarks = [
            [100, 50],
            [200, 50],
            [150, 100],
            [110, 150],
            [190, 150],
        ]
        # But testing for 180° orientation
        confidence = compute_orientation_confidence(landmarks, 180)
        assert confidence < 1.0

    def test_empty_landmarks_returns_zero_confidence(self):
        """Empty landmarks should return 0 confidence."""
        assert compute_orientation_confidence([], 0) == 0.0
        assert compute_orientation_confidence(None, 0) == 0.0


class TestRotationLandmarkTransform:
    """Tests for landmark transformation during rotation."""

    def test_180_rotation_inverts_positions(self):
        """Rotating landmarks 180° should invert positions.

        After 180° rotation, the function transforms coordinates:
        new_x = w - old_x, new_y = h - old_y

        Landmark labels (L_eye, R_eye) refer to the PERSON's left/right eye,
        not image position. The affine transform handles mapping to the
        reference template correctly without needing to swap indices.
        """
        from sim_bench.pipeline.steps.align_faces import rotate_image_and_landmarks

        # Create a simple test image (100x100)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # 5-point face landmarks: [L_eye, R_eye, nose, L_mouth, R_mouth]
        landmarks = [
            [25, 25],   # L_eye (top-left)
            [75, 25],   # R_eye (top-right)
            [50, 50],   # nose (center)
            [25, 75],   # L_mouth (bottom-left)
            [75, 75],   # R_mouth (bottom-right)
        ]

        rotated_img, rotated_lm = rotate_image_and_landmarks(image, landmarks, 180)

        # After 180° rotation - coordinate transform only (no swap):
        # [25, 25] -> [75, 75], [75, 25] -> [25, 75], [50, 50] -> [50, 50],
        # [25, 75] -> [75, 25], [75, 75] -> [25, 25]
        expected = [
            [75, 75],   # L_eye: was at top-left, now at bottom-right
            [25, 75],   # R_eye: was at top-right, now at bottom-left
            [50, 50],   # nose: center stays center
            [75, 25],   # L_mouth: was at bottom-left, now at top-right
            [25, 25],   # R_mouth: was at bottom-right, now at top-left
        ]

        for i, (actual, expect) in enumerate(zip(rotated_lm, expected)):
            assert abs(actual[0] - expect[0]) < 1e-5, f"Point {i} x mismatch: got {actual}, expected {expect}"
            assert abs(actual[1] - expect[1]) < 1e-5, f"Point {i} y mismatch: got {actual}, expected {expect}"

    def test_0_rotation_unchanged(self):
        """0° rotation should leave landmarks unchanged."""
        from sim_bench.pipeline.steps.align_faces import rotate_image_and_landmarks

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[25, 25], [75, 25], [50, 50], [25, 75], [75, 75]]

        rotated_img, rotated_lm = rotate_image_and_landmarks(image, landmarks, 0)

        assert rotated_lm == landmarks
