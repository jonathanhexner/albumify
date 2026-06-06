"""spec-079 Stage 0 — capture the Albumify golden baseline on Budapest.

Runs the Albumify pipeline (via its own API services, in-process) on
``D:\\Budapest2025_Google`` with profile_4's clustering knobs overlaid onto the
``cluster_people`` step, keeping Albumify's other params (scene clustering,
identity_refinement, selection). Snapshots the resulting ``people`` table +
scene/result summary to ``tests/_fixtures/budapest_golden/`` and prints the
identity count against the FC v2 anchor.

IMPORTANT: the FC v2 gold "15 clusters" is an IDENTITY count → compare to the
people-table row count, NOT to PipelineResult.num_clusters (that is scene/image
clustering). See specs/079-.../CLUSTERING_EXPLAINED.html.

Run:  .venv/Scripts/python scripts/capture_albumify_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from face_cluster.fc_params import FCParams  # noqa: E402
from sim_bench.pipeline.steps.configs.cluster_people import ClusterPeopleConfig  # noqa: E402
from sim_bench.api.database.session import get_session_direct  # noqa: E402
from sim_bench.api.database.models import Person  # noqa: E402
from sim_bench.api.services.album_service import AlbumService  # noqa: E402
from sim_bench.api.services.pipeline_service import PipelineService  # noqa: E402

# Anchor constants (kept in sync with tests/_budapest_baseline.py).
sys.path.insert(0, str(REPO / "tests"))
import _budapest_baseline as anchor  # noqa: E402

ALBUM_NAME = "spec079_budapest_golden"
GOLDEN_DIR = REPO / "tests" / "_fixtures" / "budapest_golden"
PIPELINE_YAML = REPO / "configs" / "pipeline.yaml"


def build_step_configs(profile_path=None) -> dict:
    """Albumify's per-step yaml config, with profile knobs overlaid on
    cluster_people (override relevant params, keep the rest)."""
    yaml_doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    steps = yaml_doc.get("default_pipeline", [])
    step_configs = {name: (yaml_doc.get(name) or {}) for name in steps}

    profile = FCParams.load(profile_path or anchor.PROFILE_PATH).model_dump()
    # Overlay ONLY the knobs cluster_people accepts (extra=forbid). FCParams
    # exposes some exemplar/attach/split knobs the step does not surface — those
    # are skipped (recorded divergence), the rest are "the relevant parameters".
    allowed = set(ClusterPeopleConfig.model_fields.keys())
    relevant = {k: v for k, v in profile.items() if k in allowed}
    skipped = sorted(set(profile) - allowed)
    cp = dict(step_configs.get("cluster_people", {}))
    cp.update(relevant)
    cp["method"] = "face_cluster_knn"
    cp["export_for_analysis"] = True
    step_configs["cluster_people"] = cp
    print(f"[capture] overlaid {len(relevant)} profile knobs; "
          f"skipped {len(skipped)} not in ClusterPeopleConfig: {skipped}")
    return step_configs


def main() -> int:
    session = get_session_direct()
    albums = AlbumService(session)
    pipeline = PipelineService(session)

    # Fresh, deterministic album (cascade-deletes any prior people/runs).
    for a in albums.list_all():
        if a.name == ALBUM_NAME:
            albums.delete(a.id)
    album = albums.create(ALBUM_NAME, str(anchor.SOURCE_DIR))
    print(f"[capture] album {album.id} ({album.image_count} images)")

    step_configs = build_step_configs()
    print(f"[capture] cluster_people K={step_configs['cluster_people'].get('K')} "
          f"blur_min={step_configs['cluster_people'].get('blur_min')}")

    job_id = pipeline.start_pipeline(
        album_id=album.id, steps=None, step_configs=step_configs, fail_fast=True,
    )
    print(f"[capture] running pipeline {job_id} ...")
    pipeline.execute_pipeline(job_id)

    run = pipeline.get_status(job_id)
    if run.status != "completed":
        print(f"[capture] FAILED: status={run.status} msg={run.error_message}")
        return 2

    people = session.query(Person).filter(Person.run_id == job_id).all()
    n_people = len(people)
    sizes = sorted((p.face_count for p in people), reverse=True)
    n_assigned = sum(p.face_count for p in people)
    result = pipeline.get_result(job_id)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / "people.json").write_text(json.dumps([
        {"person_index": p.person_index, "name": p.name,
         "face_count": p.face_count, "image_count": p.image_count,
         "thumbnail_image_path": p.thumbnail_image_path}
        for p in sorted(people, key=lambda x: -x.face_count)
    ], indent=2), encoding="utf-8")
    summary = {
        "album_id": album.id, "run_id": job_id,
        "n_people_identities": n_people,
        "n_faces_assigned": n_assigned,
        "people_sizes": sizes,
        "scene_num_clusters": result.num_clusters if result else None,
        "total_images": result.total_images if result else None,
    }
    (GOLDEN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== BUDAPEST BASELINE (Albumify, profile_4) ==========")
    print(f"  IDENTITIES (people rows) : {n_people}   <- compare to FC v2 = {anchor.EXPECTED_N_CLUSTERS}")
    print(f"  faces assigned           : {n_assigned} <- FC v2 = {anchor.EXPECTED_N_FACES_ASSIGNED}")
    print(f"  people sizes             : {sizes}")
    print(f"  FC v2 sizes              : {anchor.EXPECTED_CLUSTER_SIZES}")
    print(f"  scene num_clusters (img) : {summary['scene_num_clusters']}  (NOT the identity count)")
    try:
        anchor.assert_matches_anchor(n_people, n_assigned)
        print("  ANCHOR: PASS (within FC v2 band)")
    except AssertionError as e:
        print(f"  ANCHOR: MISS -> {e}")
    print(f"  golden saved to {GOLDEN_DIR}")
    print("=============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
