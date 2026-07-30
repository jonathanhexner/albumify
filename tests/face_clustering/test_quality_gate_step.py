"""spec-053 Phase 2 — Consolidated quality_gate step.

Three cases:
1. Step is registered under the name "quality_gate".
2. Real blur scores produced when faces have aligned crops AND blur_min > 0
   actually filter — the bug the user reported is gone.
3. Validation passes for either v2-style (face_records) or
   Albumify-style (insightface_faces + aligned_faces + face_embeddings)
   context shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

import sim_bench.pipeline.steps.all_steps  # noqa: F401 — registers steps
from sim_bench.pipeline.registry import get_registry


# ---------------------------------------------------------------------------
# Minimal context stub
# ---------------------------------------------------------------------------

@dataclass
class _FakeCtx:
    face_records: List[Any] = field(default_factory=list)
    insightface_faces: Dict[str, Any] = field(default_factory=dict)
    aligned_faces: Dict[str, Any] = field(default_factory=dict)
    face_embeddings: Dict[str, Any] = field(default_factory=dict)
    core_indices: List[int] = field(default_factory=list)
    holdout_indices: List[int] = field(default_factory=list)

    def report_progress(self, *a, **k): pass


def _aligned_low_blur() -> np.ndarray:
    """An aligned face crop with deliberately low Laplacian variance
    (uniform gray = blur_score ~= 0)."""
    return np.full((112, 112, 3), 128, dtype=np.uint8)


def _aligned_high_blur() -> np.ndarray:
    """An aligned face crop with high Laplacian variance (random noise =
    high blur_score)."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (112, 112, 3), dtype=np.uint8)


def _face(face_id: int, aligned: np.ndarray):
    from face_cluster.types import FaceRecord
    emb = np.random.randn(512).astype(np.float32)
    return FaceRecord(
        face_id=face_id, image_id=f"img{face_id}", image_path=f"img{face_id}",
        bbox=(0, 0, 100, 100), landmarks=None,
        aligned_face=aligned,
        embedding=emb,
        embedding_normalized=emb / (np.linalg.norm(emb) + 1e-9),
        pose=None, blur_score=0.0,  # will be computed by the step
        area=10000, is_core=False, face_index=face_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_step_is_registered() -> None:
    step = get_registry().get("quality_gate")
    assert step.metadata.name == "quality_gate"


def test_blur_gate_actually_filters_when_min_is_high() -> None:
    """THE bug-reproduction test. Before spec-053 the v2 step skipped
    compute_blur_scores and the gate self-disabled. With the
    consolidated step calling calc(), faces with low Laplacian variance
    must be held out when blur_min is set high."""
    ctx = _FakeCtx()
    # 3 low-blur faces (uniform gray) + 3 high-blur faces (random).
    ctx.face_records = [
        *(_face(i, _aligned_low_blur()) for i in range(3)),
        *(_face(i + 3, _aligned_high_blur()) for i in range(3)),
    ]

    step = get_registry().get("quality_gate")
    config = {
        "K": 2, "distance_threshold": 0.9, "min_cluster_size": 1,
        "blur_min": 100.0,  # high — should filter the low-blur trio
        "yaw_max": 999.0, "pitch_max": 999.0, "roll_max": 999.0,
        "max_faces_per_image_core": 50,
    }
    step.process(ctx, config)

    # Low-blur faces should be in holdout; high-blur faces in core.
    assert set(ctx.holdout_indices) >= {0, 1, 2}, (
        f"low-blur faces not held out: holdout={ctx.holdout_indices}, "
        f"core={ctx.core_indices}. The blur gate did not fire — spec-053 "
        f"regression."
    )
    # Proof that compute_blur_scores ran: the high-blur (random noise)
    # faces must have a meaningful non-zero variance. Uniform-gray faces
    # legitimately have variance == 0.0.
    high_blur_scores = [f.blur_score for f in ctx.face_records[3:]]
    assert all(s > 0.0 for s in high_blur_scores), (
        f"random-noise faces did not get a blur score — calc() didn't "
        f"call compute_blur_scores. scores={high_blur_scores}"
    )


def test_validate_accepts_v2_style_face_records() -> None:
    from face_cluster.types import FaceRecord
    ctx = _FakeCtx()
    ctx.face_records = [_face(0, _aligned_high_blur())]
    step = get_registry().get("quality_gate")
    errors = step.validate(ctx)
    assert errors == []


def test_validate_accepts_albumify_style_trio() -> None:
    ctx = _FakeCtx()
    ctx.insightface_faces = {"img0": [{"bbox": {"x_px": 0, "y_px": 0, "w_px": 10, "h_px": 10}}]}
    ctx.aligned_faces = {"img0:face_0": _aligned_high_blur()}
    ctx.face_embeddings = {"img0:face_0": np.random.randn(512).astype(np.float32)}
    step = get_registry().get("quality_gate")
    errors = step.validate(ctx)
    assert errors == []


def test_validate_rejects_when_both_input_shapes_missing() -> None:
    ctx = _FakeCtx()
    step = get_registry().get("quality_gate")
    errors = step.validate(ctx)
    assert errors and "quality_gate requires" in errors[0]
