"""spec-053 — Consolidated quality-gate step.

Replaces both ``filter_quality_gate`` (Albumify) and ``quality_gate_faces``
(FC App v2). The old steps wrapped ``QualityGater`` differently — only
one called ``compute_blur_scores`` before ``select_core_set``, so the
v2 chain silently self-disabled the blur threshold (the bug the user
reported).

This single step uses ``QualityGater.calc()`` (spec-053), which
composes the methods in the correct order by construction. The bug
class is gone.

Input compatibility: accepts EITHER
  - ``context.face_records`` (v2 path — records already built upstream), OR
  - ``context.insightface_faces`` + ``context.aligned_faces`` + ``context.face_embeddings``
    (Albumify path — this step builds records inline).

Output: writes ``core_indices``, ``holdout_indices``, ``face_records``
(with blur scores populated) to context.
"""
from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from face_cluster.config import PipelineConfig as FaceClusterConfig
from face_cluster.quality import QualityGateInputs, QualityGater
from face_cluster.types import FaceRecord
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class QualityGateStep(BaseStep):
    """Consolidated quality gate. Uses ``QualityGater.calc()``."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="quality_gate",
            display_name="Quality gate",
            description="Filter low-quality faces (blur / pose / area / det / top-k).",
            category="clustering",
            # Don't declare a strict requires set — we accept either input
            # shape (face_records OR the trio). The validate() override
            # below enforces "at least one of" semantics.
            requires=set(),
            produces={"core_indices", "holdout_indices", "face_records"},
            depends_on=[],
            config_schema={"type": "object"},
        )

    def validate(self, context: PipelineContext) -> List[str]:
        """At least one input shape must be available."""
        has_records = bool(getattr(context, "face_records", None))
        has_trio = all(
            bool(getattr(context, key, None))
            for key in ("insightface_faces", "aligned_faces", "face_embeddings")
        )
        if not has_records and not has_trio:
            return [
                "quality_gate requires either context.face_records OR "
                "(context.insightface_faces + aligned_faces + face_embeddings)."
            ]
        return []

    def process(self, context: PipelineContext, config: dict) -> None:
        start = time.time()
        faces = self._get_or_build_face_records(context)
        if not faces:
            context.core_indices = []
            context.holdout_indices = []
            context.face_records = []
            return

        fc_cfg = FaceClusterConfig(**config)
        use_pose = bool(config.get("use_pose_estimation", False))
        gater = QualityGater(fc_cfg, use_pose_estimation=use_pose)
        result = gater.calc(QualityGateInputs(faces=faces))

        # Defensive: faces with no embedding must never enter the core set.
        # (Spec-040 A1 dual-write regression guard, carried over from the
        # old quality_gate_faces step.)
        clean_core: list[int] = []
        n_dropped = 0
        for i in result.core_indices:
            if result.faces[i].embedding_normalized is None:
                n_dropped += 1
            else:
                clean_core.append(i)
        if n_dropped:
            logger.warning(
                "quality_gate: dropped %d core faces with None embedding_normalized "
                "(producer dual-write regression?)", n_dropped,
            )

        context.core_indices = clean_core
        context.holdout_indices = result.holdout_indices
        context.face_records = result.faces

        elapsed = time.time() - start
        logger.info(
            "quality_gate: %d core / %d holdout / %d total (%.2fs)",
            len(clean_core), len(result.holdout_indices), len(result.faces), elapsed,
        )

    # ------------------------------------------------------------------ private

    def _get_or_build_face_records(self, context: PipelineContext) -> List[FaceRecord]:
        """Return existing face_records, or build them from the Albumify trio."""
        existing = getattr(context, "face_records", None)
        if existing:
            return existing
        return self._build_from_insightface_trio(context)

    def _build_from_insightface_trio(
        self, context: PipelineContext,
    ) -> List[FaceRecord]:
        """Albumify path: build FaceRecord list from
        ``insightface_faces`` + ``aligned_faces`` + ``face_embeddings``."""
        face_records: List[FaceRecord] = []
        face_id = 0
        for img_path, faces_info in context.insightface_faces.items():
            for face_idx, face_info in enumerate(faces_info):
                face_key = f"{img_path}:face_{face_idx}"
                emb = context.face_embeddings.get(face_key)
                if emb is None:
                    continue
                aligned = context.aligned_faces.get(face_key)
                bbox = face_info.get("bbox", {})
                w_px, h_px = bbox.get("w_px", 0), bbox.get("h_px", 0)
                pose = face_info.get("pose")  # (yaw, pitch, roll) or None
                face_records.append(FaceRecord(
                    face_id=face_id,
                    image_id=str(img_path),
                    bbox=(
                        bbox.get("x_px", 0), bbox.get("y_px", 0),
                        bbox.get("x_px", 0) + w_px, bbox.get("y_px", 0) + h_px,
                    ),
                    landmarks=None,
                    aligned_face=aligned,
                    embedding=emb,
                    embedding_normalized=emb / (np.linalg.norm(emb) + 1e-8),
                    pose=pose,
                    blur_score=0.0,  # filled by QualityGater.calc()
                    area=w_px * h_px,
                    is_core=False,
                    image_path=str(img_path),
                    face_index=face_idx,
                ))
                face_id += 1
        logger.info("quality_gate: built %d FaceRecord(s) from Albumify trio",
                    len(face_records))
        return face_records


__all__ = ["QualityGateStep"]
