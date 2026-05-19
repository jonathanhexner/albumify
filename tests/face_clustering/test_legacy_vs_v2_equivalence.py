"""spec-040 A2 + follow-up: equivalence on real producer output, multiple configs, multiple fixtures.

Replaces the synthetic-data fixture from the original A2 landing. Two
producer-chain runs (one shared by all small-fixture tests, one for the
opt-in larger fixture) feed both clustering paths:

    detect_persons -> insightface_detect_faces -> detect_face_orientation
        -> align_faces -> extract_face_embeddings
    |
    +-- legacy: bridge run_face_cluster_knn(insightface_faces, face_embeddings)
    +-- v2:     FCAppRunner().run(face_records)         # A1 dual-write
    |
    pairwise cluster-assignment agreement >= 0.95   (the spec-040 merge gate)

Faces are matched across paths by (image_path, face_index). face_id is
not portable (legacy reconstructs faces from dicts; v2 uses the detector's
sequential ids).

Test surface:

* ``test_legacy_and_v2_agree_on_real_fixture`` — 9-jpg labeled fixture,
  parametrized over 4 configs (default, merge_on, tighter_threshold,
  larger_K). Each config is a separate spec-040 merge-gate assertion.
* ``test_legacy_and_v2_produce_same_cluster_count`` — same parametrization;
  cluster-count diff <= 1.
* ``test_legacy_and_v2_agree_on_100_image_fixture`` — opt-in slow test
  (``pytest -m slow``) on 50 jpgs from test_data/face_clustering_100/.
  Default config only (merge stage is covered by the small-fixture
  parametrization).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from face_cluster.fc_app_runner import FCAppRunner, UNIFIED_CLUSTERING_STEPS
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.cluster_people import FaceForClustering
from sim_bench.pipeline.steps.face_cluster_bridge import run_face_cluster_knn

from tests.conftest import get_test_data_dir


FIXTURE_DIR = get_test_data_dir() / "face_clustering"
FIXTURE_DIR_100 = get_test_data_dir() / "face_clustering_100"

PRODUCER_STEPS = [
    "detect_persons",
    "insightface_detect_faces",
    "detect_face_orientation",
    "align_faces",
    "extract_face_embeddings",
]

# Permissive clustering config: blur gate pinned 0 to match the bridge's
# build_fc_config pin (the InsightFace path has no blur scorer); pose gates
# permissive so faces without populated pose are not dropped. K and threshold
# chosen for a small (~9-face) real-embedding fixture.
CANONICAL_CONFIG: Dict[str, float] = {
    "K": 3,
    "distance_threshold": 0.5,
    "min_cluster_size": 2,
    "blur_min": 0.0,
    "yaw_max": 999.0,
    "pitch_max": 999.0,
    "roll_max": 999.0,
    "max_faces_per_image_core": 50,
    "merge_enabled": False,
    "attach_enabled": False,
}

# Config matrix for the small-fixture sweep. Each entry exercises a
# different part of the clustering surface; all must yield >= 0.95
# agreement between legacy and v2.
CONFIGS: List[Tuple[str, Dict]] = [
    ("default", CANONICAL_CONFIG),
    ("merge_on", {**CANONICAL_CONFIG, "merge_enabled": True}),
    ("tighter_threshold", {**CANONICAL_CONFIG, "distance_threshold": 0.35}),
    ("larger_K", {**CANONICAL_CONFIG, "K": 5}),
]


# ---------------------------------------------------------------------------
# Fixtures: producer-chain runs (one per dataset, module-scoped, single-shot)
# ---------------------------------------------------------------------------

def _pick_jpgs_by_person(n_per_person: int = 3, n_persons: int = 3) -> List[Path]:
    """Sorted jpgs from person_N/ subdirs of the labeled fixture."""
    if not FIXTURE_DIR.exists():
        return []
    picked: List[Path] = []
    for person_dir in sorted(FIXTURE_DIR.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        jpgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() == ".jpg")
        picked.extend(jpgs[:n_per_person])
        if len(picked) >= n_per_person * n_persons:
            break
    return picked


def _pick_jpgs_flat(n: int) -> List[Path]:
    """Sorted jpgs from a flat fixture dir. HEICs are skipped — InsightFace
    needs OpenCV-loadable images and HEIC support is environment-dependent
    on Windows.
    """
    if not FIXTURE_DIR_100.exists():
        return []
    jpgs = sorted(p for p in FIXTURE_DIR_100.iterdir() if p.suffix.lower() == ".jpg")
    return jpgs[:n]


def _run_producer_chain(tmp_path_factory, images: List[Path], label: str) -> PipelineContext:
    """Stage `images` into a temp dir and run the producer chain; return the context.

    Skips on missing fixture, producer failure, or empty face_records.
    """
    src_dir = tmp_path_factory.mktemp(label)
    for src in images:
        (src_dir / src.name).write_bytes(src.read_bytes())

    import sim_bench.pipeline.steps.all_steps  # noqa: F401  -- registers steps
    from sim_bench.pipeline.config import PipelineConfig
    from sim_bench.pipeline.executor import PipelineExecutor
    from sim_bench.pipeline.registry import get_registry

    context = PipelineContext(source_directory=src_dir)
    context.image_paths = sorted(src_dir.iterdir())

    step_configs = {name: {} for name in PRODUCER_STEPS}
    context.step_configs = step_configs
    config = PipelineConfig(step_configs=step_configs, fail_fast=True)

    executor = PipelineExecutor(get_registry())
    result = executor.execute(context, PRODUCER_STEPS, config=config)
    if not result.success:
        pytest.skip(
            f"Producer chain failed: {result.error_message} "
            f"(failed step: {result.failed_step}). Equivalence is "
            "unverifiable in this env."
        )
    if not context.face_records:
        pytest.skip("Producer chain ran but face_records is empty (no faces detected).")
    return context


@pytest.fixture(scope="module")
def producer_context(tmp_path_factory):
    """9-jpg labeled fixture from ``test_data/face_clustering/`` (3 persons x 3 jpgs)."""
    images = _pick_jpgs_by_person(n_per_person=3, n_persons=3)
    if not images:
        pytest.skip(
            f"Equivalence fixture missing at {FIXTURE_DIR}. "
            "Expected person_1/, person_2/, person_3/ each with .jpg files."
        )
    return _run_producer_chain(tmp_path_factory, images, "equiv_src")


@pytest.fixture(scope="module")
def producer_context_100(tmp_path_factory):
    """50-jpg flat fixture from ``test_data/face_clustering_100/`` (~5-8 min producer time)."""
    images = _pick_jpgs_flat(50)
    if not images:
        pytest.skip(
            f"Larger fixture missing at {FIXTURE_DIR_100}. "
            "Populate with: python scripts/create_test_sample.py "
            "--source D:/Google_Germany --dest test_data/face_clustering_100 "
            "--n 100 --seed 42"
        )
    return _run_producer_chain(tmp_path_factory, images, "equiv_100_src")


# ---------------------------------------------------------------------------
# Helpers: build legacy inputs from the producer context; pairwise agreement
# ---------------------------------------------------------------------------

def _build_legacy_inputs(
    context: PipelineContext,
) -> Tuple[List[FaceForClustering], np.ndarray]:
    """Reconstruct (faces, embeddings_norm) the way ClusterPeopleStep does.

    Mirrors ``ClusterPeopleStep._collect_faces_with_embeddings`` for the
    InsightFace branch, which is the branch the production pipeline takes.
    """
    faces: List[FaceForClustering] = []
    for img_path, face_data in (context.insightface_faces or {}).items():
        path_str = str(img_path).replace("\\", "/")
        for face_info in face_data.get("faces", []):
            face_index = face_info.get("face_index", 0)
            cache_key = f"{path_str}:face_{face_index}"
            embedding = context.face_embeddings.get(cache_key)
            if embedding is None:
                continue
            faces.append(FaceForClustering(
                original_path=Path(img_path),
                face_index=face_index,
                embedding=embedding,
                bbox=face_info.get("bbox", {}),
            ))
    if not faces:
        return [], np.zeros((0, 0), dtype=np.float32)
    embeddings = np.array([f.embedding for f in faces])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return faces, embeddings / norms


def _norm_path(p) -> str:
    return str(p).replace("\\", "/")


def _legacy_labels_by_identity(
    legacy_faces: List[FaceForClustering], labels_arr: np.ndarray
) -> Dict[Tuple[str, int], int]:
    return {
        (_norm_path(f.original_path), int(f.face_index)): int(labels_arr[i])
        for i, f in enumerate(legacy_faces)
    }


def _v2_labels_by_identity(
    context: PipelineContext,
) -> Dict[Tuple[str, int], int]:
    """face_records-keyed assignments; noise faces default to -1."""
    out: Dict[Tuple[str, int], int] = {}
    for cid, faces in (context.people_clusters or {}).items():
        for f in faces:
            out[(_norm_path(f.image_path), int(f.face_index))] = int(cid)
    for r in context.face_records:
        out.setdefault((_norm_path(r.image_path), int(r.face_index)), -1)
    return out


def _pairwise_agreement(
    labels_a: Dict[Tuple[str, int], int],
    labels_b: Dict[Tuple[str, int], int],
) -> float:
    """Pairwise same-cluster agreement; noise (-1) treated as singleton.

    1.0 = full agreement; 0.0 = total disagreement. Only keys present in
    both maps are scored — mismatched identity sets are a separate failure
    and would surface as a different test.
    """
    common = sorted(set(labels_a) & set(labels_b))
    if len(common) < 2:
        return 1.0
    total = 0
    agree = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            ki, kj = common[i], common[j]
            la_i, la_j = labels_a[ki], labels_a[kj]
            lb_i, lb_j = labels_b[ki], labels_b[kj]
            same_a = (la_i == la_j) and (la_i != -1)
            same_b = (lb_i == lb_j) and (lb_i != -1)
            total += 1
            if same_a == same_b:
                agree += 1
    return agree / total if total else 1.0


def _run_both_paths(
    context: PipelineContext, config: Dict
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], int], int, int]:
    """Run legacy bridge + v2 FCAppRunner on the same producer context.

    Returns (legacy_labels, v2_labels, legacy_n_clusters, v2_n_clusters).
    The contexts passed to each path are scratch instances; the shared
    ``producer_context`` is not mutated.
    """
    legacy_faces, embeddings_norm = _build_legacy_inputs(context)
    assert legacy_faces, "legacy collection produced no faces from a non-empty producer context"

    legacy_scratch = PipelineContext()
    legacy_labels_arr, *_ = run_face_cluster_knn(
        legacy_faces, embeddings_norm, config, legacy_scratch
    )
    legacy_labels = _legacy_labels_by_identity(legacy_faces, legacy_labels_arr)
    legacy_n_clusters = len({int(x) for x in legacy_labels_arr} - {-1})

    v2_ctx = PipelineContext()
    v2_ctx.face_records = list(context.face_records)
    v2_result = FCAppRunner().run(
        v2_ctx,
        step_configs={name: config for name in UNIFIED_CLUSTERING_STEPS},
    )
    assert v2_result.success, f"v2 runner failed: {v2_result.error_message}"
    v2_labels = _v2_labels_by_identity(v2_ctx)

    return legacy_labels, v2_labels, legacy_n_clusters, v2_result.n_clusters


# ---------------------------------------------------------------------------
# Tests: small-fixture sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_id,config", CONFIGS, ids=[cid for cid, _ in CONFIGS]
)
def test_legacy_and_v2_agree_on_real_fixture(producer_context, config_id, config):
    """spec-040 merge gate: real producer output -> two clustering paths -> >=95% agree.

    Same context state powers both paths (A1 dual-write enables this):
    legacy reads ``insightface_faces`` + ``face_embeddings``; v2 reads
    ``face_records``. Parametrized across 4 configs so the merge and
    high-K parts of the algorithm surface are also asserted, not just
    the default config (see REVIEW.md C5).
    """
    legacy_labels, v2_labels, _, _ = _run_both_paths(producer_context, config)

    legacy_keys = set(legacy_labels)
    v2_keys = set(v2_labels)
    assert legacy_keys == v2_keys, (
        f"[{config_id}] identity-set mismatch: "
        f"only-legacy={legacy_keys - v2_keys}, only-v2={v2_keys - legacy_keys}"
    )

    agreement = _pairwise_agreement(legacy_labels, v2_labels)
    assert agreement >= 0.95, (
        f"[{config_id}] legacy vs v2 pairwise agreement = {agreement:.3f}, "
        f"below the 0.95 bar.\n"
        f"  legacy labels: {legacy_labels}\n"
        f"  v2 labels:     {v2_labels}"
    )


@pytest.mark.parametrize(
    "config_id,config", CONFIGS, ids=[cid for cid, _ in CONFIGS]
)
def test_legacy_and_v2_produce_same_cluster_count(producer_context, config_id, config):
    """Cluster counts within +/-1 between paths across all sweep configs."""
    _, _, legacy_n, v2_n = _run_both_paths(producer_context, config)
    diff = abs(legacy_n - v2_n)
    assert diff <= 1, (
        f"[{config_id}] cluster count diverges by {diff}: "
        f"legacy={legacy_n}, v2={v2_n}"
    )


# ---------------------------------------------------------------------------
# Test: larger-fixture (opt-in via -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_legacy_and_v2_agree_on_100_image_fixture(producer_context_100):
    """Scale test: 50-jpg producer run -> >=95% pairwise agreement.

    Excluded from default test runs (~5-8 min producer chain).
    Run with ``pytest -m slow tests/face_clustering/test_legacy_vs_v2_equivalence.py``.

    Larger N tightens the pairwise-agreement signal: 50 imgs -> ~50-150
    faces -> ~1k-10k pairs (vs ~36 pairs for the small fixture). A
    same-numeric-bar (0.95) is therefore a much stronger statement of
    equivalence at this size.
    """
    legacy_labels, v2_labels, legacy_n, v2_n = _run_both_paths(
        producer_context_100, CANONICAL_CONFIG
    )

    legacy_keys = set(legacy_labels)
    v2_keys = set(v2_labels)
    assert legacy_keys == v2_keys, (
        f"identity-set mismatch: only-legacy={len(legacy_keys - v2_keys)} faces, "
        f"only-v2={len(v2_keys - legacy_keys)} faces"
    )

    agreement = _pairwise_agreement(legacy_labels, v2_labels)
    n_faces = len(legacy_keys)
    n_pairs = n_faces * (n_faces - 1) // 2
    assert agreement >= 0.95, (
        f"100-img legacy vs v2 pairwise agreement = {agreement:.4f} "
        f"({int(round(agreement * n_pairs))}/{n_pairs} pairs agree), "
        f"below the 0.95 bar. legacy_n_clusters={legacy_n}, v2_n_clusters={v2_n}"
    )
