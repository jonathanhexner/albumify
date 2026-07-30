"""Spec-102 T2 — the Albumify arm: run the real album pipeline, emit an ordered K-sequence.

Runs `sim_bench/pipeline` (the album selector: score -> penalties -> scene-cluster -> select_best)
on the shared 768px working set, then reads `context.selected_images` + `composite_scores` +
`scene_clusters`. Because select_best has no fixed-K knob, `curation.py` enforces exactly K
(coverage-first over Albumify's own picks, backfilled from the remaining scored images), then
orders the K chronologically. The scene-cluster map is persisted too — the VLM arm's contact
sheets and the duplicate-survival metric both read it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from sim_bench.albumify_vs_vlm.curation import (
    chronological_order,
    select_top_k_coverage_first,
)
from sim_bench.albumify_vs_vlm.schema import AlbumResult, Pick, save_album_result

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"

__all__ = ["AlbumifyArmConfig", "run_albumify_arm", "save_album_result"]


@dataclass
class AlbumifyArmConfig:
    target_k: int = 20
    # "default" = full 33-step product (OOMs on low-RAM machines at the occlusion CLIP load);
    # "faces"   = full people+quality+scene stack MINUS score_occlusion/score_tilt/straighten and
    #             the post-select_best face-cluster tail (everything before the OOM point);
    # "minimal" = scenes + IQA/AVA quality + select_best only (no people signal).
    pipeline: str = "faces"
    max_images_per_cluster: int = 2
    # spec-103: insert build_scene_distance before cluster_scenes (fuse visual + short-range capture
    # time). Default False = today's purely-visual scene clustering.
    fuse_scene_distance: bool = False


# Steps dropped for the "faces" pipeline: the two that OOM (occlusion CLIP + GeoCalib tilt) plus
# straighten, and the face-cluster post-processing that runs AFTER select_best (so it never
# influences the album picks). detect_persons stays -> person_penalty remains active.
_FACES_DROP = {
    "score_occlusion", "score_tilt", "straighten_images",
    "select_face_exemplars", "merge_face_clusters", "attach_holdout_faces",
    "apply_diameter_cap", "assign_people_clusters", "face_cluster_analysis_export",
    "identity_refinement", "cluster_by_identity",
}


def _resolve_steps(doc: dict, pipeline: str) -> list[str]:
    if pipeline == "faces":
        return [s for s in doc["default_pipeline"] if s not in _FACES_DROP]
    key = "default_pipeline" if pipeline == "default" else "minimal_pipeline"
    return list(doc[key])


def _insert_scene_distance(steps: list[str]) -> list[str]:
    """spec-103: place build_scene_distance immediately before cluster_scenes.

    No-op if cluster_scenes is absent or the step is already present. Returns a new list.
    """
    if "cluster_scenes" not in steps or "build_scene_distance" in steps:
        return list(steps)
    out = list(steps)
    out.insert(out.index("cluster_scenes"), "build_scene_distance")
    return out


def _stem_of(path: str, straightened_from: dict[str, str]) -> str:
    """Map a pipeline pick back to its working-set stem (undoing any straighten derivative)."""
    orig = straightened_from.get(path, path)
    return Path(orig).stem


def run_albumify_arm(
    imgs_dir: Path,
    trip: str,
    input_set_hash: str,
    config: AlbumifyArmConfig | None = None,
) -> AlbumResult:
    """Run the album pipeline on `imgs_dir` (the 768px working set) and return the K-sequence."""
    cfg = config or AlbumifyArmConfig()
    # Imports are lazy: they pull in the heavy step registry (InsightFace/YOLO/etc.) only when
    # the arm actually runs, keeping `import curation` cheap for tests.
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.run import execute_spec
    from sim_bench.pipeline.spec import PipelineSpec

    doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    steps = _resolve_steps(doc, cfg.pipeline)
    if cfg.fuse_scene_distance:
        steps = _insert_scene_distance(steps)  # spec-103: fuse visual + short-range capture time
    step_configs = {name: (doc.get(name) or {}) for name in steps}
    # Enforce the per-cluster policy we reason about (select_best default is 2).
    step_configs.setdefault("select_best", {})
    step_configs["select_best"] = {**step_configs["select_best"],
                                   "max_images_per_cluster": cfg.max_images_per_cluster}

    spec = PipelineSpec(steps=steps, step_configs=step_configs)
    context = PipelineContext(source_directory=imgs_dir)

    logger.info("running Albumify arm (%s pipeline) on %s", cfg.pipeline, imgs_dir)
    result = execute_spec(spec, context, fail_fast=True)
    if not result.success:
        raise RuntimeError(f"Albumify pipeline failed at {result.failed_step}: "
                           f"{result.error_message}")

    straightened_from = getattr(context, "straightened_from", {}) or {}
    scores = {
        _stem_of(p, straightened_from): float(s)
        for p, s in (context.composite_scores or {}).items()
    }
    labels = {
        _stem_of(p, straightened_from): int(c)
        for p, c in (context.scene_cluster_labels or {}).items()
    }
    scene_clusters = {
        int(cid): [_stem_of(p, straightened_from) for p in paths]
        for cid, paths in (context.scene_clusters or {}).items()
    }
    selected = [_stem_of(p, straightened_from) for p in (context.selected_images or [])]

    # Coverage-first to K over Albumify's own picks; backfill from remaining scored images.
    chosen = select_top_k_coverage_first(selected, scores, labels, cfg.target_k)
    if len(chosen) < cfg.target_k:
        chosen_set = set(chosen)
        remaining = [s for s in scores if s not in chosen_set]
        chosen += select_top_k_coverage_first(
            remaining, scores, labels, cfg.target_k - len(chosen)
        )

    order = chronological_order(chosen)
    picks = [
        Pick(id=s, score=round(scores.get(s, 0.0), 4), scene_cluster=labels.get(s, -1),
             reason=f"composite={scores.get(s, 0.0):.3f}")
        for s in order
    ]
    logger.info("Albumify arm: %d picks (target K=%d), %d scene clusters",
                len(order), cfg.target_k, len(scene_clusters))
    return AlbumResult(
        trip=trip, arm="albumify", input_set_hash=input_set_hash, k=cfg.target_k,
        pipeline=cfg.pipeline, order=order, picks=picks, scene_clusters=scene_clusters,
        # Full per-image composite scores — EXP-2 needs the argmax within each scene cluster
        # (Albumify's best-of-cluster), which the K=20 picks alone can't recover.
        meta={"composite_scores": {s: round(v, 4) for s, v in scores.items()}},
    )
