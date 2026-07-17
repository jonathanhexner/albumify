"""spec-079 — prove the clustering aligns, without touching persistence.

Runs ALBUMIFY's producer steps (its default_pipeline up to extract_face_embeddings)
followed by FC v2's UNIFIED clustering steps, fed profile_5 via to_step_configs().
Counts context.people_clusters (= assign_people_clusters output, pre-refinement)
and compares to the FC v2 reference (8).

If this prints 8, Albumify's faces + FC v2's clustering == FC v2. The remaining
work to make the *app* show 8 is then only the people-table shape adapter
(FaceRecord vs FaceForClustering), not the clustering itself.

Run:  .venv/Scripts/python scripts/prove_albumify_unified.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import _budapest_baseline as anchor  # noqa: E402
from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS  # noqa: E402
from face_cluster.fc_params import FCParams  # noqa: E402
from sim_bench.pipeline.context import PipelineContext  # noqa: E402
from sim_bench.pipeline.run import execute_spec  # noqa: E402
from sim_bench.pipeline.spec import PipelineSpec  # noqa: E402

PIPELINE_YAML = REPO / "configs" / "pipeline.yaml"


def main() -> int:
    doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    default_pipeline = doc["default_pipeline"]
    # Albumify producer = everything before the clustering/persistence steps.
    producer = default_pipeline[: default_pipeline.index("cluster_people")]

    steps = list(producer) + list(UNIFIED_CLUSTERING_STEPS)
    step_configs = {name: (doc.get(name) or {}) for name in producer}
    step_configs.update(FCParams.load(anchor.PROFILE_PATH).to_step_configs())

    spec = PipelineSpec(steps=steps, step_configs=step_configs)
    print(f"[prove] producer steps ({len(producer)}): {producer}")
    print(f"[prove] + unified clustering ({len(UNIFIED_CLUSTERING_STEPS)})")
    print(f"[prove] profile: {anchor.PROFILE_PATH.name}")

    context = PipelineContext(source_directory=anchor.SOURCE_DIR)
    result = execute_spec(spec, context, fail_fast=True)
    if not result.success:
        print(f"[prove] FAILED at {result.failed_step}: {result.error_message}")
        return 2

    people = getattr(context, "people_clusters", {}) or {}
    sizes = sorted((len(v) for v in people.values()), reverse=True)
    n_faces = len(getattr(context, "face_records", []) or [])
    n_assigned = sum(sizes)

    print("\n========== ALBUMIFY producer + FC v2 clustering ==========")
    print(f"  faces (face_records) : {n_faces}")
    print(f"  identities           : {len(people)}   <- FC v2 reference = {anchor.EXPECTED_N_CLUSTERS}")
    print(f"  assigned             : {n_assigned}    <- FC v2 = {anchor.EXPECTED_N_FACES_ASSIGNED}")
    print(f"  sizes                : {sizes}")
    print(f"  FC v2 sizes          : {anchor.EXPECTED_CLUSTER_SIZES}")
    verdict = "MATCH (8=8)" if len(people) == anchor.EXPECTED_N_CLUSTERS else "STILL DIFFERS"
    print(f"  VERDICT              : {verdict}")
    print("==========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
