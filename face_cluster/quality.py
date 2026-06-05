"""Quality gating for face clustering - blur and pose filtering."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from face_cluster.types import FaceRecord, GateResult, QualityVerdict
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


# spec-053: typed boundary for QualityGater.calc(). Inputs/Result are
# deliberately small — NOT a PipelineContext. The pipeline step is the
# only place that knows both worlds.

@dataclass(frozen=True, slots=True)
class QualityGateInputs:
    """Per-call data for QualityGater.calc()."""
    faces: List[FaceRecord]


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    """Output of QualityGater.calc().

    ``faces`` is the SAME list passed in, mutated in place with computed
    blur scores (and optionally pose) before gating. Callers should
    treat this as authoritative for the post-gating face state.
    """
    core_indices: List[int]
    holdout_indices: List[int]
    verdicts: List[QualityVerdict]
    faces: List[FaceRecord]


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
        # spec-041 follow-up — symmetric to the pose gate's vacuous-pass.
        # When the producer chain doesn't populate blur_score (it's 0.0 for
        # every face), select_core_set will bypass blur_min for the run
        # and emit a WARNING. ``None`` means "use self.config.blur_min".
        self._effective_blur_min: Optional[float] = None

    def calc(self, inputs: QualityGateInputs) -> QualityGateResult:
        """Single pipeline entry point (spec-053).

        Composes ``compute_blur_scores → [compute_pose_scores] →
        select_core_set`` in the correct order. Pipeline steps MUST
        use this method rather than calling the individual methods —
        that's how the spec-053 ``filter_quality_gate`` /
        ``quality_gate_faces`` divergence (one forgot to compute blur)
        becomes impossible by construction.

        The individual methods stay public for notebook callers that
        need finer control.
        """
        faces = self.compute_blur_scores(inputs.faces)
        if self.use_pose_estimation:
            faces = self.compute_pose_scores(faces)
        core, holdout, verdicts = self.select_core_set(faces)
        return QualityGateResult(
            core_indices=list(core),
            holdout_indices=list(holdout),
            verdicts=verdicts,
            faces=faces,
        )

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
        # spec-041 follow-up #3: tell the truth about the pose gate's
        # current state. The InsightFace producer chain does NOT populate
        # FaceRecord.pose today — no `extract_face_pose` step exists yet.
        # With pose universally None, this gate can only REJECT (when
        # require_pose=True) or pass vacuously (when False). The thresholds
        # are inert until a producer step lands. Tracked as a sighting.
        n_with_pose = sum(1 for f in faces if f.pose is not None)
        if n_with_pose == 0:
            logger.warning(
                "Pose gate: NOT WIRED — face.pose is None for all %d faces "
                "(no producer step extracts yaw/pitch/roll today). With "
                "require_pose=%s, the gate will %s every face. "
                "Thresholds yaw<=%s pitch<=%s roll<=%s are inert until an "
                "extract_face_pose step exists.",
                len(faces), self.config.require_pose,
                "REJECT" if self.config.require_pose else "vacuously pass",
                self.config.yaw_max, self.config.pitch_max, self.config.roll_max,
            )
        else:
            logger.info(
                "Pose gate: ACTIVE on %d/%d faces (yaw<=%s, pitch<=%s, "
                "roll<=%s, require_pose=%s)",
                n_with_pose, len(faces),
                self.config.yaw_max, self.config.pitch_max,
                self.config.roll_max, self.config.require_pose,
            )

        # spec-041 follow-up — same diagnosis as pose, but for blur. The
        # InsightFace producer chain has no blur scorer, so every face's
        # ``blur_score`` is the default 0.0. With ``blur_min=50.0`` (the
        # FCConfig default) the gate then rejects every face. Detect the
        # condition, log loudly, and override ``blur_min`` to 0 for this
        # run so the threshold is inert when there's no data — matching
        # the pose gate's vacuous-pass behavior. Tracked as SIGHTING-068.
        n_with_blur = sum(1 for f in faces if f.blur_score > 0.0)
        if n_with_blur == 0 and self.config.blur_min > 0:
            logger.warning(
                "Blur gate: NOT WIRED — face.blur_score is 0.0 for all %d "
                "faces (no producer step computes blur today). Threshold "
                "blur_min=%s would reject every face; bypassing it for "
                "this run. Add an insightface_score_blur step to enable.",
                len(faces), self.config.blur_min,
            )
            self._effective_blur_min = 0.0
        else:
            self._effective_blur_min = self.config.blur_min

        core_indices: List[int] = []
        holdout_indices: List[int] = []
        verdicts: List[QualityVerdict] = []
        # spec-041 follow-up — per-gate rejection counters. When 0 candidates
        # survive, emit a WARNING with the breakdown so the user sees which
        # gate caused the failure without grepping per-face debug logs.
        # Tracked as SIGHTING-069.
        gate_rejections: Dict[str, int] = {}

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
                for gate_name, gate_result in verdict.gates.items():
                    if not gate_result.passed:
                        gate_rejections[gate_name] = gate_rejections.get(gate_name, 0) + 1

        logger.info(
            f"Quality gating: {len(core_indices)} core, "
            f"{len(holdout_indices)} holdout faces"
        )

        # If everything got rejected, dump the per-gate breakdown so the
        # post-mortem is one log line, not a code archaeology session.
        if len(core_indices) == 0 and len(candidate_set) > 0:
            n_candidates = len(candidate_set)
            breakdown = ", ".join(
                f"{name}={count}/{n_candidates}"
                for name, count in sorted(gate_rejections.items(), key=lambda kv: -kv[1])
                if count > 0
            ) or "(none — top-K-per-image dropped them before gate evaluation)"
            logger.warning(
                "Quality gate rejected ALL %d candidates. Per-gate rejection "
                "(faces failing each gate; a face may fail multiple): %s",
                n_candidates, breakdown,
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
        self._add_area_pct_gate(face, gates)
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
        self._add_area_pct_gate(face, gates)

        # Priority order for rejection_reason (det_score first — most fundamental)
        _priority = ("det_score", "blur", "pose_yaw", "pose_pitch", "area", "area_pct")
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
        # spec-041 follow-up — use the effective (run-time) blur threshold.
        # When the producer chain doesn't populate blur_score for any face,
        # select_core_set sets _effective_blur_min=0.0 so the gate passes
        # vacuously instead of rejecting everything. See SIGHTING-068.
        threshold = (
            self._effective_blur_min
            if self._effective_blur_min is not None
            else self.config.blur_min
        )
        gates["blur"] = GateResult(
            value=face.blur_score,
            threshold=threshold,
            passed=face.blur_score >= threshold,
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

    def _add_area_pct_gate(self, face: FaceRecord, gates: Dict[str, GateResult]) -> None:
        """Gate on face bbox area as a % of the image (spec-073).

        Resolution-independent complement to the pixel ``area`` gate. Skipped
        when ``min_face_area_pct`` is None (disabled). Permissive (passes) when
        the face has no ``area_ratio`` — don't penalise legacy runs that
        predate the detection-time ratio."""
        thr = getattr(self.config, "min_face_area_pct", None)
        if thr is None:
            return
        ratio = getattr(face, "area_ratio", None)
        if ratio is None:
            gates["area_pct"] = GateResult(value=-1.0, threshold=thr, passed=True)
        else:
            pct = float(ratio) * 100.0
            gates["area_pct"] = GateResult(value=pct, threshold=thr, passed=pct >= thr)
