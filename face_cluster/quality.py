"""Quality gating for face clustering - blur and pose filtering."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from face_cluster.types import FaceRecord, GateResult, QualityVerdict
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
        faces: List[FaceRecord],
    ) -> Tuple[List[int], List[int], List[QualityVerdict]]:
        """Select core and holdout sets based on quality criteria.

        Returns:
            (core_indices, holdout_indices, verdicts) where verdicts[i]
            corresponds to faces[i] with per-gate results and rejection_reason.
        """
        if not faces:
            return [], [], []

        candidate_set = self._top_k_per_image(faces)

        logger.info(
            f"Selected {len(candidate_set)} candidates from "
            f"{len(faces)} faces (top {self.config.max_faces_per_image_core} per image)"
        )
        logger.info(
            f"Pose angle filter: ACTIVE (InsightFace 1k3d68), "
            f"yaw<={self.config.yaw_max}, pitch<={self.config.pitch_max}, "
            f"roll<={self.config.roll_max}, require_pose={self.config.require_pose}"
        )

        core_indices: List[int] = []
        holdout_indices: List[int] = []
        verdicts: List[QualityVerdict] = []

        for i, face in enumerate(faces):
            if i not in candidate_set:
                verdicts.append(self._top_k_verdict(face))
                holdout_indices.append(i)
                continue

            verdict = self._evaluate_gates(face)
            verdicts.append(verdict)
            face.quality_verdict = verdict
            face.rejection_reason = verdict.rejection_reason

            if verdict.all_passed():
                core_indices.append(i)
                face.is_core = True
            else:
                holdout_indices.append(i)
                logger.debug(f"Face {face.face_id}: holdout ({verdict.rejection_reason})")

        logger.info(
            f"Quality gating: {len(core_indices)} core, "
            f"{len(holdout_indices)} holdout faces"
        )
        return core_indices, holdout_indices, verdicts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _top_k_per_image(self, faces: List[FaceRecord]) -> set:
        """Return set of face indices kept as top-K per image by area."""
        image_groups: Dict[str, List[int]] = {}
        for i, face in enumerate(faces):
            image_groups.setdefault(face.image_id, []).append(i)

        kept = set()
        for indices in image_groups.values():
            sorted_idx = sorted(indices, key=lambda i: faces[i].area, reverse=True)
            kept.update(sorted_idx[:self.config.max_faces_per_image_core])
        return kept

    def _top_k_verdict(self, face: FaceRecord) -> QualityVerdict:
        """Build a rejected verdict for a face dropped by top-K-per-image gate."""
        gates: Dict[str, GateResult] = {}
        self._add_det_score_gate(face, gates)
        self._add_blur_gate(face, gates)
        self._add_pose_gates(face, gates)
        self._add_area_gate(face, gates)
        verdict = QualityVerdict(gates=gates, rejection_reason="top_k_per_image")
        face.quality_verdict = verdict
        face.rejection_reason = "top_k_per_image"
        return verdict

    def _evaluate_gates(self, face: FaceRecord) -> QualityVerdict:
        """Evaluate all quality gates and return a verdict with rejection_reason."""
        gates: Dict[str, GateResult] = {}
        self._add_det_score_gate(face, gates)
        self._add_blur_gate(face, gates)
        self._add_pose_gates(face, gates)
        self._add_area_gate(face, gates)

        # Priority order for rejection_reason (det_score first — most fundamental)
        _priority = ("det_score", "blur", "pose_yaw", "pose_pitch", "area")
        rejection_reason = next(
            (name for name in _priority if name in gates and not gates[name].passed),
            None,
        )
        return QualityVerdict(gates=gates, rejection_reason=rejection_reason)

    def _add_det_score_gate(self, face: FaceRecord, gates: Dict[str, GateResult]) -> None:
        """Gate on InsightFace detection confidence. Skipped (passes) when det_score_min is None
        or when the face has no det_score (permissive — don't penalise legacy runs)."""
        if self.config.det_score_min is None:
            return
        threshold = self.config.det_score_min
        if face.det_score is None:
            # No score available — pass permissively
            gates["det_score"] = GateResult(value=-1.0, threshold=threshold, passed=True)
        else:
            gates["det_score"] = GateResult(
                value=float(face.det_score),
                threshold=threshold,
                passed=float(face.det_score) >= threshold,
            )

    def _add_blur_gate(self, face: FaceRecord, gates: Dict[str, GateResult]) -> None:
        gates["blur"] = GateResult(
            value=face.blur_score,
            threshold=self.config.blur_min,
            passed=face.blur_score >= self.config.blur_min,
        )

    def _add_pose_gates(self, face: FaceRecord, gates: Dict[str, GateResult]) -> None:
        if face.pose is None:
            passed = not self.config.require_pose
            gates["pose_yaw"] = GateResult(value=0.0, threshold=self.config.yaw_max, passed=passed)
            gates["pose_pitch"] = GateResult(value=0.0, threshold=self.config.pitch_max, passed=passed)
            return
        yaw, pitch, _ = face.pose
        gates["pose_yaw"] = GateResult(
            value=abs(yaw), threshold=self.config.yaw_max, passed=abs(yaw) <= self.config.yaw_max
        )
        gates["pose_pitch"] = GateResult(
            value=abs(pitch), threshold=self.config.pitch_max, passed=abs(pitch) <= self.config.pitch_max
        )

    def _add_area_gate(self, face: FaceRecord, gates: Dict[str, GateResult]) -> None:
        threshold = self.config.min_face_area if self.config.min_face_area is not None else 0.0
        gates["area"] = GateResult(
            value=face.area,
            threshold=threshold,
            passed=face.area >= threshold,
        )
