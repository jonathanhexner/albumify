"""spec-079 / SIGHTING-100 — identity_refinement ordering + face-type regression.

Guards the fix for the Albumify identity over-attachment. Root cause (all
fallout from the spec-079 unification that replaced the monolithic
`cluster_people` step with the 8-step chain ending in `assign_people_clusters`):

  1. ORDERING: `identity_refinement`/`cluster_by_identity`/`select_best_per_person`
     declared `depends_on=["cluster_people"]`. With that step removed, the
     executor (which orders by dependencies, not list order) had no constraint
     forcing them after clustering, so `identity_refinement` ran BEFORE
     `assign_people_clusters` on a raw pre-assignment blob → 238-face mega-cluster.
     Fixed by adding `assign_people_clusters` to each `depends_on`.
  2. PATH KEY: `_face_key` read `face.original_path`; the unified chain feeds
     `FaceRecord` (attr `image_path`). Fixed with a type-tolerant `_face_path`.
  3. MUTATION: `face.cluster_id = ...` on a frozen `FaceRecord` (extra=forbid).
     Guarded (membership lives in the refined_clusters dict anyway).

With the fixes, on Budapest + profile_5 the step receives the clean
[26,20,12,7,3,2,2] cores + 38 noise and attaches only a handful of genuinely
close leftover faces (no mega-cluster).

Run:  .venv/Scripts/python -m pytest -m budapest \
        tests/pipeline/test_identity_refinement_overattach_budapest.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.budapest

SRC = Path(r"D:\Budapest2025_Google")
PROFILE = Path.home() / ".sim_bench" / "profiles_v2" / "profile_5.json"
REPO = Path(__file__).resolve().parents[2]
PIPELINE_YAML = REPO / "configs" / "pipeline.yaml"

REFERENCE_CORE_SIZES = [26, 20, 12, 7, 3, 2, 2]
MAX_SANE_CLUSTER = 40   # a 122-photo album has no 200-face "person"


def _core_sizes(people_clusters) -> list[int]:
    from sim_bench.pipeline.clustering_labels import is_noise
    return sorted((len(v) for cid, v in people_clusters.items() if not is_noise(cid)),
                  reverse=True)


@pytest.fixture(scope="module")
def refinement_run():
    """Run the production Albumify spec up to (incl.) identity_refinement."""
    if not SRC.exists():
        pytest.skip(f"Budapest source missing: {SRC}")
    if not PROFILE.exists():
        pytest.skip(f"profile_5 missing: {PROFILE}")

    from face_cluster.fc_params import FCParams
    from sim_bench.pipeline.cache_handler import UniversalCacheHandler
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.spec import PipelineSpec
    from sim_bench.pipeline.run import execute_spec
    from sim_bench.api.database.session import get_session_direct
    from sim_bench.api.services.pipeline_service import PipelineService

    params = FCParams.load(PROFILE)
    doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    steps = list(doc.get("default_pipeline", []))
    step_configs = {name: (doc.get(name) or {}) for name in steps}
    cp = dict(doc.get("cluster_people", {}))
    cp.update(params.model_dump())
    cp["method"] = "face_cluster_knn"
    step_configs["cluster_people"] = cp

    session = get_session_direct()
    svc = PipelineService(session)
    step_configs = svc._broadcast_clustering_config(steps, dict(step_configs))

    cut = steps.index("identity_refinement") + 1
    spec = PipelineSpec(steps=steps[:cut], step_configs=step_configs)
    ctx = PipelineContext(source_directory=SRC, cache_handler=UniversalCacheHandler(session))
    execute_spec(spec, ctx, fail_fast=True)
    return ctx


def test_shared_chain_matches_reference(refinement_run):
    """The shared 8-step clustering chain yields the FC v2 reference clusters."""
    sizes = _core_sizes(refinement_run.people_clusters or {})
    assert sizes == REFERENCE_CORE_SIZES, (
        f"shared chain core clusters {sizes} != reference {REFERENCE_CORE_SIZES}"
    )


def test_identity_refinement_runs_after_clustering(refinement_run):
    """Ordering regression: identity_refinement must see the assigned clusters
    (proven by it producing a refined set), not run early on a raw blob."""
    refined = refinement_run.refined_people_clusters or {}
    assert refined, "identity_refinement produced no refined clusters"
    diag = getattr(refinement_run, "refinement_attach_diagnostics", None)
    assert diag is not None, "attach diagnostic missing"
    # It must operate on the real noise pool (38), not a degenerate 0/huge pool.
    assert diag["noise_pool"] == 38, f"unexpected noise pool: {diag}"


def test_identity_refinement_does_not_overattach(refinement_run):
    """Regression for the 238-face mega-cluster. After the fix the step rescues
    only a few genuinely-close leftover faces; no cluster balloons."""
    refined = refinement_run.refined_people_clusters or {}
    sizes = sorted((len(v) for v in refined.values()), reverse=True)
    diag = refinement_run.refinement_attach_diagnostics
    print(f"\n  refined sizes: {sizes}  attach diag: {diag}")
    assert max(sizes) <= MAX_SANE_CLUSTER, (
        f"mega-cluster regression: biggest cluster {max(sizes)} > {MAX_SANE_CLUSTER}; "
        f"sizes={sizes}"
    )
    assert len(refined) >= 6, f"clusters collapsed: {sizes}"
    assert diag["attached"] <= 15, f"over-attachment: {diag['attached']} faces attached"
