"""spec-079 Stage 0b — localize WHERE the bridge (20) and unified steps (10) diverge.

Both clustering recipes are run on the SAME face_records and the SAME FCConfig
(profile_5_nogates, so the configs are behaviorally identical). We dump the
cluster/face count after each clustering sub-stage and diff the two columns.
The first sub-stage whose counts differ is the culprit.

If the columns are identical end-to-end, the 10-vs-20 gap is NOT in the
clustering recipe — it is the INPUT face set (producer chain: 340 vs 337).

Run:  .venv/Scripts/python scripts/localize_gap.py <run_dir_with_face_records>
Default run dir: the nogates FC v2 run (340 faces).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from face_cluster.attach import HoldoutAttacher  # noqa: E402
from face_cluster.clustering import ConnectedComponentsClusterer  # noqa: E402
from face_cluster.exemplars import D10ExemplarSelector  # noqa: E402
from face_cluster.fc_params import FCParams  # noqa: E402
from face_cluster.knn_graph import KNNGraphBuilder  # noqa: E402
from face_cluster.merge import ConservativeMerger  # noqa: E402
from face_cluster.quality import QualityGater  # noqa: E402
from face_cluster.fc_app_runner import FCAppRunner  # noqa: E402
from sim_bench.run_db.store import RunStore  # noqa: E402
from sim_bench.pipeline.context import PipelineContext  # noqa: E402

PROFILE = REPO / "specs" / "079-albumify-shared-core" / "profile_5_nogates.json"
DEFAULT_RUN = Path.home() / ".sim_bench" / "runs" / "2f8d1db8057d432eb12bb720402a29ff"


def _nc(cr) -> int:
    return len(cr.clusters) if cr is not None else -1


def _assigned(cr) -> int:
    return sum(len(v) for v in cr.clusters.values()) if cr is not None else -1


def bridge_recipe(face_records, fc_cfg) -> dict:
    """The exact sub-stage sequence inside face_cluster_bridge.run_face_cluster_knn."""
    out = {"faces_in": len(face_records)}
    core, holdout, _ = QualityGater(fc_cfg).select_core_set(face_records)
    out["core_set"] = len(core)
    graph = KNNGraphBuilder(fc_cfg).build_graph(face_records, core)
    base = ConnectedComponentsClusterer(fc_cfg).cluster(graph, core)
    out["base_clusters"] = base.n_clusters
    base, _ = D10ExemplarSelector(fc_cfg).select_exemplars(base, graph)
    cr = copy.deepcopy(base)
    if fc_cfg.merge_enabled:
        cr, _, _ = ConservativeMerger(fc_cfg).merge_clusters_with_logging(cr, graph)
    out["post_merge"] = cr.n_clusters
    if fc_cfg.attach_enabled and holdout:
        cr = HoldoutAttacher(fc_cfg).attach_holdouts(face_records, core, holdout, cr, graph)
    out["post_attach"] = _nc(cr)
    out["assigned"] = _assigned(cr)
    return out


def unified_recipe(face_records, step_configs) -> dict:
    """The 8-step unified chain via FCAppRunner; read per-stage context attrs."""
    ctx = PipelineContext()
    ctx.face_records = copy.deepcopy(face_records)
    res = FCAppRunner().run(ctx, step_configs=step_configs)
    base = getattr(ctx, "cluster_result", None)
    merged = getattr(ctx, "merged_cluster_result", None)
    return {
        "faces_in": len(face_records),
        "core_set": len(getattr(ctx, "core_indices", []) or []),
        "base_clusters": _nc(base),
        "post_merge": _nc(merged) if merged is not None else _nc(base),
        "post_attach": res.n_clusters,
        "assigned": res.n_faces_assigned,
    }


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    params = FCParams.load(PROFILE)
    fc_cfg = params.to_fc_config()
    step_configs = params.to_step_configs()

    face_records = RunStore(run_dir).faces()
    print(f"run_dir   : {run_dir.name}")
    print(f"profile   : {PROFILE.name}  (merge={fc_cfg.merge_enabled})")
    print(f"faces     : {len(face_records)}\n")

    b = bridge_recipe(face_records, fc_cfg)
    u = unified_recipe(face_records, step_configs)

    stages = ["faces_in", "core_set", "base_clusters", "post_merge", "post_attach", "assigned"]
    print(f"{'SUB-STAGE':<16}{'unified':>10}{'bridge':>10}   verdict")
    print("-" * 52)
    first_div = None
    for s in stages:
        mark = "" if u[s] == b[s] else "  <-- DIVERGES"
        if u[s] != b[s] and first_div is None:
            first_div = s
        print(f"{s:<16}{u[s]:>10}{b[s]:>10}{mark}")
    print("-" * 52)
    if first_div:
        print(f"FIRST DIVERGENCE: {first_div}  ->  the culprit sub-stage.")
    else:
        print("NO divergence on identical input ->  the gap is the INPUT face set "
              "(producer chain), not the clustering recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
