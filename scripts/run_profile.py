"""Run the FC App v2 pipeline from a saved profile JSON — config in, run out.

The whole point: the Streamlit v2 app saves a profile (an ``FCParams`` JSON);
this script takes that file + a source dir and reproduces the run headlessly,
landing an app-visible run dir under ``~/.sim_bench/runs/`` so you can open the
result in the FC v2 app afterwards. No hand-written clustering params — the
profile IS the config.

Usage::

    .venv/Scripts/python scripts/run_profile.py \
        --profile ~/.sim_bench/profiles_v2/profile_5.json \
        --src "D:/Budapest2025_Google"

Optional:
    --album NAME     label recorded in action_log (default: source dir name)
    --expect-sizes "26,20,12,7,3,2,2"   assert reproduced cluster sizes (Step 0)

Exit: 0 on success (and match if --expect-sizes given), 1 on failure/mismatch.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from app.face_clustering_v2.pipeline import PRODUCER_STEPS  # noqa: E402
from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS  # noqa: E402
from face_cluster.fc_params import FCParams  # noqa: E402
from face_cluster.run_layout import allocate_run_dir  # noqa: E402
from sim_bench.pipeline.run import run_pipeline  # noqa: E402
from sim_bench.pipeline.spec import PipelineSpec  # noqa: E402

# FC v2 = discovery + producer chain + the unified clustering steps, all as ONE
# config-driven step list submitted to the one runner.
FC_V2_PRODUCER = ["discover_images"] + list(PRODUCER_STEPS)

logger = logging.getLogger("run_profile")


def cluster_sizes(db_path: Path) -> list[int]:
    """Final-iteration assigned cluster sizes, descending, from a run DB."""
    if not db_path.is_file():
        return []
    c = sqlite3.connect(str(db_path))
    try:
        mx = c.execute("SELECT MAX(iteration) FROM cluster_assignments").fetchone()[0]
        rows = c.execute(
            "SELECT cluster_id, COUNT(*) FROM cluster_assignments "
            "WHERE iteration=? AND cluster_id>=0 GROUP BY cluster_id ORDER BY 2 DESC",
            (mx,),
        ).fetchall()
        return [r[1] for r in rows]
    finally:
        c.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run the FC v2 pipeline from a profile JSON.")
    ap.add_argument("--profile", type=Path, required=True, help="FCParams profile JSON.")
    ap.add_argument("--src", type=Path, required=True, help="Source image directory.")
    ap.add_argument("--album", type=str, default=None, help="Album label (default: src dir name).")
    ap.add_argument("--expect-sizes", type=str, default=None,
                    help='Comma-separated cluster sizes to assert, e.g. "26,20,12,7,3,2,2".')
    args = ap.parse_args()

    if not args.profile.is_file():
        logger.error("Profile not found: %s", args.profile)
        return 1
    if not args.src.is_dir():
        logger.error("Source dir not found: %s", args.src)
        return 1

    params = FCParams.load(args.profile)
    album = args.album or args.src.name
    runs_base = Path.home() / ".sim_bench" / "runs"
    run_dir, run_id = allocate_run_dir(runs_base, album)

    logger.info("profile=%s  src=%s  album=%s", args.profile.name, args.src, album)
    logger.info("run_id=%s  run_dir=%s", run_id, run_dir)
    logger.info("key knobs: K=%s merge=%s blur_min=%s split=%s attach=%s",
                params.K, params.merge_enabled, params.blur_min,
                params.split_enabled, params.attach_enabled)

    spec = PipelineSpec.from_fcparams(
        params, producer_steps=FC_V2_PRODUCER, clustering_steps=UNIFIED_CLUSTERING_STEPS,
    )
    result = run_pipeline(
        source_dir=args.src, run_dir=run_dir, run_id=run_id, album=album,
        spec=spec, producer="fc_app",
    )
    if not result.success:
        logger.error("Run FAILED: %s", result.error_message)
        return 1

    sizes = cluster_sizes(result.db_path)
    print("\n========== RUN COMPLETE ==========")
    print(f"  run_id     : {run_id}")
    print(f"  db_path    : {result.db_path}")
    print(f"  images     : {result.n_images}")
    print(f"  faces      : {result.n_faces}")
    print(f"  clusters   : {result.n_clusters}  (noise={result.n_noise})")
    print(f"  sizes      : {sizes}")

    if args.expect_sizes:
        expected = [int(x) for x in args.expect_sizes.split(",") if x.strip()]
        if sizes == expected:
            print(f"  STEP 0     : PASS  (matches reference {expected})")
        else:
            print(f"  STEP 0     : MISMATCH")
            print(f"               expected {expected}")
            print(f"               got      {sizes}")
            print("==================================")
            return 1
    print("==================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
