"""Unit tests for face alignment.

Tests that the align_face functions correctly align faces
using 5-point affine transformation with orientation correction.
"""

import pytest
import numpy as np
import cv2

from sim_bench.pipeline.steps.align_faces import (
    rotate_image_and_landmarks,
    align_face_with_orientation,
)
from sim_bench.pipeline.utils.face_alignment import ARCFACE_REF_POINTS_112


class TestRotateImageAndLandmarks:
    """Tests for rotate_image_and_landmarks function."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample 200x300 image (H x W)."""
        return np.zeros((200, 300, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_landmarks(self):
        """Sample landmarks for a face in the image."""
        return [
            [100, 50],   # left_eye
            [200, 50],   # right_eye
            [150, 100],  # nose
            [110, 150],  # left_mouth
            [190, 150],  # right_mouth
        ]

    def test_0_degree_rotation_unchanged(self, sample_image, sample_landmarks):
        """0° rotation should not modify image or landmarks."""
        rotated_img, rotated_lm = rotate_image_and_landmarks(
            sample_image, sample_landmarks, 0
        )

        assert rotated_img.shape == sample_image.shape
        assert rotated_lm == sample_landmarks

    def test_90_degree_rotation_dimensions(self, sample_image, sample_landmarks):
        """90° CW rotation should swap width and height."""
        rotated_img, rotated_lm = rotate_image_and_landmarks(
            sample_image, sample_landmarks, 90
        )

        # Original: (200, 300, 3) -> After 90° CW: (300, 200, 3)
        assert rotated_img.shape == (300, 200, 3)

    def test_180_degree_rotation_dimensions(self, sample_image, sample_landmarks):
        """180° rotation should keep same dimensions."""
        rotated_img, rotated_lm = rotate_image_and_landmarks(
            sample_image, sample_landmarks, 180
        )

        assert rotated_img.shape == sample_image.shape

    def test_270_degree_rotation_dimensions(self, sample_image, sample_landmarks):
        """270° rotation should swap width and height."""
        rotated_img, rotated_lm = rotate_image_and_landmarks(
            sample_image, sample_landmarks, 270
        )

        # Original: (200, 300, 3) -> After 270°: (300, 200, 3)
        assert rotated_img.shape == (300, 200, 3)

    def test_90_degree_landmark_transform(self):
        """Test landmark transformation for 90° CW rotation."""
        # Simple 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[25, 25], [75, 25]]  # Two points

        rotated_img, rotated_lm = rotate_image_and_landmarks(image, landmarks, 90)

        # For 90° CW: new_x = h - old_y, new_y = old_x
        # Point [25, 25]: new_x = 100 - 25 = 75, new_y = 25
        # Point [75, 25]: new_x = 100 - 25 = 75, new_y = 75
        assert abs(rotated_lm[0][0] - 75) < 1e-5
        assert abs(rotated_lm[0][1] - 25) < 1e-5
        assert abs(rotated_lm[1][0] - 75) < 1e-5
        assert abs(rotated_lm[1][1] - 75) < 1e-5

    def test_180_degree_landmark_transform(self):
        """Test landmark transformation for 180° rotation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[25, 25], [75, 25]]

        rotated_img, rotated_lm = rotate_image_and_landmarks(image, landmarks, 180)

        # For 180°: new_x = w - old_x, new_y = h - old_y
        # Point [25, 25]: new_x = 100 - 25 = 75, new_y = 100 - 25 = 75
        # Point [75, 25]: new_x = 100 - 75 = 25, new_y = 100 - 25 = 75
        assert abs(rotated_lm[0][0] - 75) < 1e-5
        assert abs(rotated_lm[0][1] - 75) < 1e-5
        assert abs(rotated_lm[1][0] - 25) < 1e-5
        assert abs(rotated_lm[1][1] - 75) < 1e-5

    def test_270_degree_landmark_transform(self):
        """Test landmark transformation for 270° rotation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[25, 25], [75, 25]]

        rotated_img, rotated_lm = rotate_image_and_landmarks(image, landmarks, 270)

        # For 270°: new_x = old_y, new_y = w - old_x
        # Point [25, 25]: new_x = 25, new_y = 100 - 25 = 75
        # Point [75, 25]: new_x = 25, new_y = 100 - 75 = 25
        assert abs(rotated_lm[0][0] - 25) < 1e-5
        assert abs(rotated_lm[0][1] - 75) < 1e-5
        assert abs(rotated_lm[1][0] - 25) < 1e-5
        assert abs(rotated_lm[1][1] - 25) < 1e-5


class TestAlignFaceWithOrientation:
    """Tests for align_face_with_orientation function."""

    @pytest.fixture
    def face_image(self):
        """Create a 400x400 image with a synthetic 'face'."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw circles at landmark positions for visibility
        # Eyes at (120, 100) and (280, 100)
        cv2.circle(img, (120, 100), 10, (255, 255, 255), -1)
        cv2.circle(img, (280, 100), 10, (255, 255, 255), -1)
        # Nose at (200, 200)
        cv2.circle(img, (200, 200), 10, (255, 255, 255), -1)
        # Mouth at (140, 300) and (260, 300)
        cv2.circle(img, (140, 300), 8, (255, 255, 255), -1)
        cv2.circle(img, (260, 300), 8, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def upright_landmarks(self):
        """Landmarks for an upright face."""
        return [
            [120, 100],  # left_eye
            [280, 100],  # right_eye
            [200, 200],  # nose
            [140, 300],  # left_mouth
            [260, 300],  # right_mouth
        ]

    @pytest.fixture
    def upside_down_landmarks(self):
        """Landmarks for an upside-down face (eyes at bottom)."""
        return [
            [120, 300],  # left_eye (at bottom)
            [280, 300],  # right_eye (at bottom)
            [200, 200],  # nose (above eyes)
            [140, 100],  # left_mouth (at top)
            [260, 100],  # right_mouth (at top)
        ]

    def test_align_upright_face_produces_output(self, face_image, upright_landmarks):
        """Alignment of upright face should produce valid output."""
        aligned = align_face_with_orientation(
            face_image, upright_landmarks, orientation_angle=0, target_size=256
        )

        assert aligned is not None
        assert aligned.shape == (256, 256, 3)

    def test_align_upside_down_face_produces_output(self, face_image, upside_down_landmarks):
        """Alignment of upside-down face with 180° correction should produce output."""
        aligned = align_face_with_orientation(
            face_image, upside_down_landmarks, orientation_angle=180, target_size=256
        )

        assert aligned is not None
        assert aligned.shape == (256, 256, 3)

    def test_alignment_without_orientation_correction(self, face_image, upside_down_landmarks):
        """Aligning upside-down face without correction should still work (but poorly)."""
        # This tests the fallback behavior - alignment works but face is wrong
        aligned = align_face_with_orientation(
            face_image, upside_down_landmarks, orientation_angle=0, target_size=256
        )

        assert aligned is not None
        # The face will be upside down in the output, but function should not crash

    def test_target_size_respected(self, face_image, upright_landmarks):
        """Output should match target_size."""
        for size in [112, 160, 256, 512]:
            aligned = align_face_with_orientation(
                face_image, upright_landmarks, orientation_angle=0, target_size=size
            )
            assert aligned.shape == (size, size, 3)


class TestArcFaceReferencePoints:
    """Tests for ArcFace reference template."""

    def test_reference_points_shape(self):
        """Reference points should be 5x2 array."""
        assert ARCFACE_REF_POINTS_112.shape == (5, 2)

    def test_reference_points_are_positive(self):
        """All reference points should be positive."""
        assert np.all(ARCFACE_REF_POINTS_112 >= 0)

    def test_reference_points_within_112(self):
        """Reference points should be within 112x112."""
        assert np.all(ARCFACE_REF_POINTS_112 <= 112)

    def test_eyes_above_mouth_in_reference(self):
        """Eyes should be above mouth in reference template."""
        left_eye_y = ARCFACE_REF_POINTS_112[0, 1]
        right_eye_y = ARCFACE_REF_POINTS_112[1, 1]
        left_mouth_y = ARCFACE_REF_POINTS_112[3, 1]
        right_mouth_y = ARCFACE_REF_POINTS_112[4, 1]

        eye_center_y = (left_eye_y + right_eye_y) / 2
        mouth_center_y = (left_mouth_y + right_mouth_y) / 2

        # In image coordinates, y increases downward
        # Eyes above mouth means eyes have smaller y
        assert eye_center_y < mouth_center_y, "Eyes should be above mouth in reference"

    def test_nose_between_eyes_and_mouth(self):
        """Nose should be between eyes and mouth in reference."""
        nose_y = ARCFACE_REF_POINTS_112[2, 1]
        eye_center_y = (ARCFACE_REF_POINTS_112[0, 1] + ARCFACE_REF_POINTS_112[1, 1]) / 2
        mouth_center_y = (ARCFACE_REF_POINTS_112[3, 1] + ARCFACE_REF_POINTS_112[4, 1]) / 2

        assert eye_center_y < nose_y < mouth_center_y, "Nose should be between eyes and mouth"
