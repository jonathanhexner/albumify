"""Quality gating for face clustering - blur and pose filtering."""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from face_cluster.types import FaceRecord
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estimate head pose using SixDRepNet (optional dependency)."""

    def __init__(self, device: str = 'cpu'):
        """Initialize pose estimator.

        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load SixDRepNet model."""
        if self._model is not None:
            return

        try:
            from sixdrepnet import SixDRepNet
            gpu_id = -1 if self.device == 'cpu' else 0
            self._model = SixDRepNet(gpu_id=gpu_id)
            logger.info(f"SixDRepNet model loaded (device={self.device})")
        except ImportError:
            logger.warning("SixDRepNet not installed. Install with: pip install sixdrepnet")
            self._model = None

    def estimate_pose(self, face_crop: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Estimate pose from face crop.

        Args:
            face_crop: Face image (HxWx3 RGB uint8)

        Returns:
            (yaw, pitch, roll) in degrees, or None if estimation fails
        """
        if self._model is None:
            self._load_model()

        if self._model is None:
            return None

        try:
            # Convert to PIL Image
            if face_crop.shape[2] == 3:
                image = Image.fromarray(face_crop)
            else:
                image = Image.fromarray(face_crop[:, :, :3])

            # Convert to BGR for SixDRepNet
            img_bgr = np.array(image)[:, :, ::-1]

            # Predict returns (pitch, yaw, roll)
            pitch, yaw, roll = self._model.predict(img_bgr)

            return (float(yaw[0]), float(pitch[0]), float(roll[0]))

        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return None


class QualityGater:
    """Quality gating for face selection.

    Filters faces based on:
    - Blur score (Laplacian variance)
    - Pose angles (yaw/pitch/roll)
    - Face area
    - Top-K faces per image
    """

    def __init__(self, config: PipelineConfig, use_pose_estimation: bool = False, device: str = 'cpu'):
        """Initialize quality gater.

        Args:
            config: Pipeline configuration with quality thresholds
            use_pose_estimation: Whether to estimate pose from face crops (requires sixdrepnet)
            device: Device for pose estimation ('cpu' or 'cuda')
        """
        self.config = config
        self.use_pose_estimation = use_pose_estimation
        self.pose_estimator = PoseEstimator(device) if use_pose_estimation else None

    def compute_blur_scores(self, faces: List[FaceRecord]) -> List[FaceRecord]:
        """Compute blur scores for all faces.

        Uses variance of Laplacian on aligned face crop.

        Args:
            faces: List of FaceRecord objects

        Returns:
            Updated faces with blur_score set
        """
        for face in faces:
            if face.aligned_face is None:
                logger.warning(f"No aligned face for face_id {face.face_id}")
                face.blur_score = 0.0
                continue

            # Resize to 112x112 if not already
            img = face.aligned_face
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112))

            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Compute Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            face.blur_score = variance

        logger.info(f"Computed blur scores for {len(faces)} faces")
        return faces

    def compute_pose_scores(self, faces: List[FaceRecord]) -> List[FaceRecord]:
        """Compute pose (yaw/pitch/roll) for all faces using SixDRepNet.

        Args:
            faces: List of FaceRecord objects with aligned_face set

        Returns:
            Updated faces with pose set
        """
        if not self.use_pose_estimation or self.pose_estimator is None:
            logger.warning("Pose estimation not enabled or SixDRepNet not available")
            return faces

        logger.info(f"Computing pose for {len(faces)} faces...")

        for face in faces:
            if face.aligned_face is None:
                logger.debug(f"No aligned face for face_id {face.face_id}")
                continue

            pose = self.pose_estimator.estimate_pose(face.aligned_face)
            if pose is not None:
                face.pose = pose
            else:
                logger.debug(f"Pose estimation failed for face_id {face.face_id}")

        poses_computed = sum(1 for f in faces if f.pose is not None and f.pose != (0.0, 0.0, 0.0))
        logger.info(f"Computed pose for {poses_computed}/{len(faces)} faces")

        return faces

    def select_core_set(
        self,
        faces: List[FaceRecord]
    ) -> Tuple[List[int], List[int]]:
        """Select core and holdout sets based on quality criteria.

        Selection process:
        1. Group faces by image_id
        2. Keep only top max_faces_per_image_core by area per image
        3. Apply quality filters:
           - abs(yaw) <= yaw_max
           - abs(pitch) <= pitch_max
           - abs(roll) <= roll_max
           - blur_score >= blur_min
           - area >= min_face_area (if set)
        4. Faces passing all filters go to core, others to holdout

        Args:
            faces: List of FaceRecord objects

        Returns:
            (core_indices, holdout_indices) tuple
        """
        if not faces:
            return [], []

        # Group by image_id
        image_groups = {}
        for i, face in enumerate(faces):
            if face.image_id not in image_groups:
                image_groups[face.image_id] = []
            image_groups[face.image_id].append(i)

        # Keep only top K faces per image by area
        candidate_indices = []
        for image_id, indices in image_groups.items():
            # Sort by area descending
            indices_sorted = sorted(indices, key=lambda i: faces[i].area, reverse=True)
            # Keep top max_faces_per_image_core
            top_k = indices_sorted[:self.config.max_faces_per_image_core]
            candidate_indices.extend(top_k)

        logger.info(
            f"Selected {len(candidate_indices)} candidates from "
            f"{len(faces)} faces (top {self.config.max_faces_per_image_core} per image)"
        )

        # Pose angles come from InsightFace's 1k3d68 model — reliable for gating.
        # Gating is always active; require_pose controls whether faces with no pose data
        # (detection failed) are sent to holdout.
        apply_pose_angles = True
        logger.info(
            f"Pose angle filter: ACTIVE (InsightFace 1k3d68), "
            f"yaw<={self.config.yaw_max}, pitch<={self.config.pitch_max}, "
            f"roll<={self.config.roll_max}, require_pose={self.config.require_pose}"
        )

        core_indices = []
        holdout_indices = []

        for i, face in enumerate(faces):
            # Check if this face is a candidate
            if i not in candidate_indices:
                holdout_indices.append(i)
                continue

            # Check quality criteria
            passes_all = True

            # Pose check — only when calibrated pose available
            if apply_pose_angles:
                if face.pose is None:
                    if self.config.require_pose:
                        passes_all = False
                        logger.debug(f"Face {face.face_id}: no pose data, require_pose=True -> holdout")
                else:
                    yaw, pitch, roll = face.pose
                    if abs(yaw) > self.config.yaw_max:
                        passes_all = False
                        logger.debug(f"Face {face.face_id}: yaw {yaw:.1f} exceeds {self.config.yaw_max}")
                    if abs(pitch) > self.config.pitch_max:
                        passes_all = False
                        logger.debug(f"Face {face.face_id}: pitch {pitch:.1f} exceeds {self.config.pitch_max}")
                    if abs(roll) > self.config.roll_max:
                        passes_all = False
                        logger.debug(f"Face {face.face_id}: roll {roll:.1f} exceeds {self.config.roll_max}")

            # Blur check
            if face.blur_score < self.config.blur_min:
                passes_all = False
                logger.debug(
                    f"Face {face.face_id}: blur {face.blur_score:.1f} "
                    f"below {self.config.blur_min}"
                )

            # Area check
            if self.config.min_face_area is not None:
                if face.area < self.config.min_face_area:
                    passes_all = False
                    logger.debug(
                        f"Face {face.face_id}: area {face.area:.0f} "
                        f"below {self.config.min_face_area}"
                    )

            if passes_all:
                core_indices.append(i)
                face.is_core = True
            else:
                holdout_indices.append(i)

        logger.info(
            f"Quality gating: {len(core_indices)} core, "
            f"{len(holdout_indices)} holdout faces"
        )

        return core_indices, holdout_indices
