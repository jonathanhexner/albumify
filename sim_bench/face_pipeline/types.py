"""Data types for face processing pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class BoundingBox:
    """Face bounding box in both relative and absolute coordinates."""
    # Relative coordinates (0-1 range)
    x: float  # Left edge
    y: float  # Top edge
    w: float  # Width
    h: float  # Height

    # Absolute pixel coordinates (set after image dimensions known)
    x_px: int = 0
    y_px: int = 0
    w_px: int = 0
    h_px: int = 0

    def to_absolute(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Convert relative to absolute pixel coordinates."""
        return BoundingBox(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            x_px=int(self.x * img_width),
            y_px=int(self.y * img_height),
            w_px=int(self.w * img_width),
            h_px=int(self.h * img_height)
        )

    @property
    def area_ratio(self) -> float:
        """Face area as ratio of image area (relative coords)."""
        return self.w * self.h

    @property
    def center(self) -> tuple:
        """Center point (relative coordinates)."""
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class PoseEstimate:
    """Head pose estimation result from SixDRepNet."""
    yaw: float    # Left-right rotation in degrees (-90 to +90)
    pitch: float  # Up-down rotation in degrees (-90 to +90)
    roll: float   # Tilt rotation in degrees (-90 to +90)

    # Thresholds for "frontal" face
    YAW_THRESHOLD: float = field(default=20.0, repr=False)
    PITCH_THRESHOLD: float = field(default=15.0, repr=False)
    ROLL_THRESHOLD: float = field(default=10.0, repr=False)

    @property
    def is_frontal(self) -> bool:
        """Check if pose is within acceptable frontal range."""
        return (
            abs(self.yaw) <= self.YAW_THRESHOLD and
            abs(self.pitch) <= self.PITCH_THRESHOLD and
            abs(self.roll) <= self.ROLL_THRESHOLD
        )

    def compute_penalty(self, steepness: float = 0.2) -> float:
        """
        Compute soft sigmoid penalty for non-frontal poses.

        Uses sigmoid function for smooth transition around thresholds.
        Returns 0.0 for perfect frontal, approaches 1.0 for extreme angles.

        Args:
            steepness: How quickly penalty increases past threshold (default 0.2)
        """
        def sigmoid_penalty(angle: float, threshold: float) -> float:
            """Sigmoid penalty that's ~0 below threshold, ~1 well above."""
            # Shifted sigmoid: 0.5 at threshold, approaches 1 as angle increases
            deviation = abs(angle) - threshold
            if deviation <= 0:
                return 0.0
            return 1.0 / (1.0 + np.exp(-steepness * deviation))

        yaw_penalty = sigmoid_penalty(self.yaw, self.YAW_THRESHOLD)
        pitch_penalty = sigmoid_penalty(self.pitch, self.PITCH_THRESHOLD)
        roll_penalty = sigmoid_penalty(self.roll, self.ROLL_THRESHOLD)

        # Max of all penalties (worst angle determines penalty)
        return max(yaw_penalty, pitch_penalty, roll_penalty)

    @property
    def frontal_score(self) -> float:
        """Score from 0-1 where 1 is perfectly frontal."""
        return 1.0 - self.compute_penalty()


@dataclass
class FaceQualityScore:
    """Combined face quality assessment."""
    pose_score: float       # 0-1, higher = more frontal
    eyes_open_score: float  # 0-1, higher = more open
    smile_score: float      # 0-1, higher = more smile
    sharpness_score: float  # 0-1, technical quality
    detection_confidence: float  # MediaPipe detection confidence

    # Weights for overall score
    POSE_WEIGHT: float = field(default=0.3, repr=False)
    EYES_WEIGHT: float = field(default=0.3, repr=False)
    SMILE_WEIGHT: float = field(default=0.2, repr=False)
    SHARPNESS_WEIGHT: float = field(default=0.2, repr=False)

    @property
    def overall(self) -> float:
        """Weighted combination of all quality factors."""
        return (
            self.POSE_WEIGHT * self.pose_score +
            self.EYES_WEIGHT * self.eyes_open_score +
            self.SMILE_WEIGHT * self.smile_score +
            self.SHARPNESS_WEIGHT * self.sharpness_score
        )


@dataclass
class CroppedFace:
    """A cropped face extracted from an image."""
    # Source information
    original_path: Path
    face_index: int  # Index if multiple faces in image

    # Cropped image (PIL format for easy manipulation)
    image: Image.Image

    # Bounding box
    bbox: BoundingBox

    # Detection info
    detection_confidence: float
    face_ratio: float  # Face area / image area

    # Path to saved cropped face image (if saved to disk)
    crop_path: Optional[Path] = None

    # Quality metrics (filled in by quality scorer)
    pose: Optional[PoseEstimate] = None
    eyes_open_score: Optional[float] = None
    smile_score: Optional[float] = None
    is_smiling: Optional[bool] = None
    both_eyes_open: Optional[bool] = None

    # Embedding (filled in by embedder)
    embedding: Optional[np.ndarray] = None

    # Overall quality
    quality: Optional[FaceQualityScore] = None

    # Clustering result
    cluster_id: Optional[int] = None

    @property
    def crop_key(self) -> str:
        """Unique key for this cropped face."""
        return f"{self.original_path}:face_{self.face_index}"


@dataclass
class FaceCluster:
    """A cluster of faces belonging to same identity."""
    cluster_id: int
    faces: List[CroppedFace]

    # Computed properties
    centroid: Optional[np.ndarray] = None
    representative_face: Optional[CroppedFace] = None

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_images(self) -> int:
        """Number of unique source images containing this person."""
        return len(set(f.original_path for f in self.faces))

    def compute_centroid(self):
        """Compute centroid embedding for cluster."""
        embeddings = [f.embedding for f in self.faces if f.embedding is not None]
        if embeddings:
            self.centroid = np.mean(embeddings, axis=0)
            # Normalize to unit length
            self.centroid = self.centroid / np.linalg.norm(self.centroid)

    def select_representative(self):
        """Select best quality face as cluster representative."""
        faces_with_quality = [f for f in self.faces if f.quality is not None]
        if faces_with_quality:
            self.representative_face = max(faces_with_quality, key=lambda f: f.quality.overall)
        elif self.faces:
            self.representative_face = self.faces[0]


@dataclass
class AlbumFaceResult:
    """Complete face processing result for an album."""
    # All detected faces
    all_faces: List[CroppedFace]

    # Identity clusters
    clusters: List[FaceCluster]

    # Images without usable faces
    images_without_faces: List[Path]

    # Processing stats
    total_images: int
    images_with_faces: int
    total_faces_detected: int
    faces_meeting_threshold: int  # Faces >= 2% of image

    @property
    def num_identities(self) -> int:
        return len(self.clusters)

    def get_faces_for_image(self, image_path: Path) -> List[CroppedFace]:
        """Get all faces detected in a specific image."""
        return [f for f in self.all_faces if f.original_path == image_path]

    def get_images_for_cluster(self, cluster_id: int) -> List[Path]:
        """Get all images containing a specific person."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return list(set(f.original_path for f in cluster.faces))
        return []


# Constants
MIN_FACE_RATIO = 0.02  # 2% of image area minimum
