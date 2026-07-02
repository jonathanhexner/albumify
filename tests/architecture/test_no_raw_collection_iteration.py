"""spec-032 P5: forbid raw iteration of filterable collections.

Downstream pipeline steps must consume ``ctx.filters.active(item_type)`` so
filter decisions actually gate behavior. Iterating raw collections like
``context.image_paths`` or ``context.insightface_faces`` silently bypasses
the filter contract — this is the bug pattern from SIGHTING-059.

This check enforces the rule via grep. Locked decision per spec-032:
initial allow-list grandfathers every existing step that violates. P5+
shrinks the list one PR at a time as each step migrates. CI prevents NEW
violations from any future step author.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# Directories scanned. Tests + spec docs are excluded.
SCAN_DIRS = [
    "sim_bench/pipeline/steps",
    "face_cluster",
    "app",
]


# Forbidden patterns: regex matched against each non-empty, non-comment line.
# Each pattern has a human-readable label used in failure output.
FORBIDDEN = [
    ("context.image_paths iteration",
     re.compile(r"for\s+\w+\s+in\s+context\.image_paths\b")),
    ("context.insightface_faces iteration",
     re.compile(r"for\s+\w+\s+in\s+context\.insightface_faces\b")),
    ("ctx.faces iteration",
     re.compile(r"for\s+\w+\s+in\s+ctx\.faces\b")),
    ("context.faces iteration",
     re.compile(r"for\s+\w+\s+in\s+context\.faces\b")),
    ("context.quality_passed iteration",
     re.compile(r"for\s+\w+\s+in\s+context\.quality_passed\b")),
    ("context.active_images iteration",
     re.compile(r"for\s+\w+\s+in\s+context\.active_images\b")),
]


# Files allowed to iterate raw collections — each entry needs a reason.
# Per spec-032 locked decision: initial allow-list grandfathers existing
# state; subsequent PRs shrink the list as each file migrates.
ALLOW_LIST: dict[str, str] = {
    # Producers — they CREATE the items, so they have nothing to filter against.
    "sim_bench/pipeline/steps/discover_images.py":
        "producer of image_paths",
    "sim_bench/pipeline/steps/score_iqa.py":
        "scoring producer; filter_quality consumes the scores",
    "sim_bench/pipeline/steps/score_ava.py":
        "scoring producer",
    "sim_bench/pipeline/steps/score_quality.py":
        "scoring producer (spec-093 multi-method IQA over all images)",
    "sim_bench/pipeline/steps/extract_geo_metadata.py":
        "producer of geo_metadata (EXIF per image, spec-094)",
    "sim_bench/pipeline/steps/infer_geo_clip.py":
        "producer of geo_clip_predictions (StreetCLIP per image)",
    "sim_bench/pipeline/steps/infer_geo_coords.py":
        "producer of geo_coord_predictions (GeoCLIP per image)",
    "sim_bench/pipeline/steps/caption_images.py":
        "producer of image_captions (BLIP per image)",
    "sim_bench/pipeline/steps/score_face_quality.py":
        "scoring producer for per-face quality",
    "sim_bench/pipeline/steps/extract_scene_embedding.py":
        "producer of scene embeddings; runs on active_images conceptually",
    "sim_bench/pipeline/steps/extract_face_embeddings.py":
        "producer of face embeddings",
    "sim_bench/pipeline/steps/insightface_detect_faces.py":
        "producer of insightface_faces — the source of truth",
    "sim_bench/pipeline/steps/detect_faces.py":
        "producer (MediaPipe path)",
    "sim_bench/pipeline/steps/detect_persons.py":
        "producer of person detections",
    "sim_bench/pipeline/steps/score_face_eyes.py":
        "scoring producer (eye state)",
    "sim_bench/pipeline/steps/score_face_smile.py":
        "scoring producer (smile)",
    "sim_bench/pipeline/steps/score_face_pose.py":
        "scoring producer (pose)",
    "sim_bench/pipeline/steps/score_face_frontal.py":
        "scoring producer (frontal score)",
    "sim_bench/pipeline/steps/insightface_score_expression.py":
        "scoring producer (expression)",
    "sim_bench/pipeline/steps/insightface_score_eyes.py":
        "scoring producer (eyes)",
    "sim_bench/pipeline/steps/insightface_score_pose.py":
        "scoring producer (pose)",
    "sim_bench/pipeline/steps/detect_face_orientation.py":
        "producer (orientation)",
    "sim_bench/pipeline/steps/align_faces.py":
        "producer (aligned faces)",
    "sim_bench/pipeline/steps/validate_alignment.py":
        "diagnostic — reads raw faces for sampling",
    "sim_bench/pipeline/steps/crop_faces.py":
        "producer (raw bbox crops, debug)",
    "sim_bench/pipeline/steps/save_face_debug_artifacts.py":
        "debug artifact writer",
    "sim_bench/pipeline/steps/build_knn_graph.py":
        "operates on filter_quality_gate output (core_indices)",
    "sim_bench/pipeline/steps/identity_refinement.py":
        "operates on people_clusters output",
    "sim_bench/pipeline/steps/cluster_by_identity.py":
        "operates on face_clusters output",
    "sim_bench/pipeline/steps/cluster_people.py":
        "operates on face_embeddings (filter_faces output)",
    "sim_bench/pipeline/steps/cluster_scenes.py":
        "operates on scene_embeddings",
    "sim_bench/pipeline/steps/cluster_connected_components.py":
        "operates on graph",
    "sim_bench/pipeline/steps/compute_debug_distances.py":
        "diagnostic — reads raw clusters",
    "sim_bench/pipeline/steps/export_for_labeling.py":
        "exporter — reads final results",
    "sim_bench/pipeline/steps/filter_best_faces.py":
        "TODO P5 migration: filter that should emit decisions",
    "sim_bench/pipeline/steps/filter_faces.py":
        "TODO P5 migration: legacy filter, still writes face['filter_passed']",
    "sim_bench/pipeline/steps/filter_portraits.py":
        "TODO P5 migration: filter that should emit decisions",
    "sim_bench/pipeline/steps/quality_gate.py":
        "spec-053: consolidated step (was filter_quality_gate + quality_gate_faces)",
    "sim_bench/pipeline/steps/select_best.py":
        "selection step (post-filter)",
    "sim_bench/pipeline/steps/select_best_per_person.py":
        "selection step (post-filter)",
    "sim_bench/pipeline/steps/select_exemplars.py":
        "exemplar selection (post-filter)",
    "sim_bench/pipeline/steps/face_cluster_bridge.py":
        "bridge from Albumify to face_cluster — operates on filtered list",
    "sim_bench/pipeline/steps/face_cluster_export.py":
        "exporter (post-pipeline)",
    "face_cluster/pipeline.py":
        "FC App orchestrator; calls ctx.filters internally but also iterates "
        "ctx.faces for stage book-keeping",
    "face_cluster/quality.py":
        "TODO P5 migration: emits decisions via verdicts; bridged in pipeline._record_quality_filters",
    "face_cluster/embedding.py":
        "embedding producer",
    "face_cluster/crops.py":
        "crop producer",
    "face_cluster/export.py":
        "legacy exporter (spec-030 Phase 4 will remove)",
    "face_cluster/result_db.py":
        "legacy DB writer (spec-030 Phase 4 will remove)",
    "face_cluster/loader.py":
        "loader (read path, not a filter)",
    "face_cluster/manual_merge_snapshot.py":
        "manual merge writer",
    "face_cluster/merge.py":
        "cluster-level operator (operates on cluster results)",
    "sim_bench/run_db/exporter.py":
        "exporter (spec-030, post-pipeline; relocated by spec-056)",
    "face_cluster/exemplars.py":
        "exemplar selection (post-filter)",
    "face_cluster/cluster_diameter_cap.py":
        "post-merge cap (spec-031); operates on cluster results",
    "face_cluster/knn_graph.py":
        "graph builder; takes core_indices already filtered",
    "face_cluster/clustering.py":
        "connected components on graph (post-filter)",
}


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_no, label, matching_line)] for forbidden patterns in path."""
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return violations
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for label, pat in FORBIDDEN:
            if pat.search(line):
                violations.append((i, label, stripped[:120]))
    return violations


def test_no_raw_collection_iteration_outside_allow_list():
    all_violations: list[str] = []
    for d in SCAN_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(REPO)).replace("\\", "/")
            if rel in ALLOW_LIST:
                continue
            for line_no, label, snippet in _scan_file(path):
                all_violations.append(
                    f"{rel}:{line_no}  [{label}]  {snippet}"
                )
    assert not all_violations, (
        "Raw collection iteration detected outside ALLOW_LIST.  Either:\n"
        "  (a) migrate this step to consume ctx.filters.active(item_type), or\n"
        "  (b) add the file to ALLOW_LIST with a one-line reason (reviewer-gated).\n"
        "Violations:\n  "
        + "\n  ".join(all_violations)
    )


def test_allow_list_entries_have_non_empty_reasons():
    bad = [k for k, v in ALLOW_LIST.items() if not v.strip()]
    assert not bad, (
        "ALLOW_LIST entries must include a reason:\n  "
        + "\n  ".join(bad)
    )


def test_allow_list_files_exist():
    """Stale allow-list entries (files deleted or moved) fail CI.

    Prevents the list from accumulating dead weight as steps are refactored.
    """
    missing = [k for k in ALLOW_LIST if not (REPO / k).exists()]
    assert not missing, (
        "ALLOW_LIST references files that no longer exist — remove them:\n  "
        + "\n  ".join(missing)
    )
