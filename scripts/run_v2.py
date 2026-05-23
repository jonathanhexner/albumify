"""spec-041 — headless CLI runner for the FC App v2 pipeline.

Drives ``app.face_clustering_v2.pipeline.run_v2_pipeline`` from the shell.
Lets you reproduce a UI run, drive equivalence sweeps from a Makefile, or
script A/B comparisons against the legacy bridge — no Streamlit required.

Examples
--------
Run from a profile JSON::

    .venv/Scripts/python scripts/run_v2.py \\
        --src D:/Google_Germany \\
        --out runs/v2_smoke \\
        --profile ~/.sim_bench/profiles/baseline.json

Run with inline overrides (Tier-1 knobs only — full set goes via ``--profile``)::

    .venv/Scripts/python scripts/run_v2.py --src D:/album --out runs/v2_tight \\
        --K 3 --distance_threshold 0.30 --merge

Save the resolved params back to a profile JSON::

    .venv/Scripts/python scripts/run_v2.py --src D:/album --out runs/v2_save \\
        --K 5 --save-profile profiles/k5.json

Exit codes: 0 on success, 1 on pipeline failure, 2 on argument / IO error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Make `app.*` and `face_cluster.*` importable when invoked as a script
# from anywhere (the script lives at <repo>/scripts/, repo root is parent).
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import ValidationError  # noqa: E402

from app.face_clustering_v2.pipeline import run_v2_pipeline  # noqa: E402
from face_cluster.fc_params import FCParams  # noqa: E402
from sim_bench.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _build_params(args: argparse.Namespace) -> FCParams:
    """Profile (if given) is the base; explicit flags override its values."""
    if args.profile is not None:
        params = FCParams.load(args.profile)
    else:
        params = FCParams()

    overrides = {}
    for name in ("K", "distance_threshold", "min_cluster_size",
                 "blur_min", "max_faces_per_image_core",
                 "yaw_max", "pitch_max", "roll_max"):
        v = getattr(args, name, None)
        if v is not None:
            overrides[name] = v
    if args.merge:
        overrides["merge_enabled"] = True
    if args.cap:
        overrides["cluster_diameter_cap_enabled"] = True
    if overrides:
        params = params.model_copy(update=overrides)
    return params


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="Source image directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--profile", type=Path, default=None,
                        help="FCParams profile JSON to load as the base configuration.")
    parser.add_argument("--save-profile", type=Path, default=None,
                        help="If set, save the resolved FCParams to this path before running.")

    # Tier-1 inline overrides.
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--distance_threshold", type=float, default=None)
    parser.add_argument("--min_cluster_size", type=int, default=None)
    parser.add_argument("--blur_min", type=float, default=None)
    parser.add_argument("--max_faces_per_image_core", type=int, default=None)
    parser.add_argument("--yaw_max", type=float, default=None)
    parser.add_argument("--pitch_max", type=float, default=None)
    parser.add_argument("--roll_max", type=float, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="Shorthand for merge_enabled=True.")
    parser.add_argument("--cap", action="store_true",
                        help="Shorthand for cluster_diameter_cap_enabled=True.")

    args = parser.parse_args(argv)
    # spec-041 follow-up: align logging across surfaces. Writes to
    # logs/<timestamp>/cli_run_v2.log alongside Albumify / FC apps.
    setup_logging("cli_run_v2")

    if not args.src.exists():
        logger.error("Source directory does not exist: %s", args.src)
        return 2

    try:
        params = _build_params(args)
    except (ValidationError, OSError) as e:
        logger.error("Could not build FCParams: %s", e)
        return 2

    if args.save_profile is not None:
        try:
            params.save(args.save_profile)
            logger.info("Saved profile to %s", args.save_profile)
        except OSError as e:
            logger.error("Could not save profile: %s", e)
            return 2

    logger.info("Running v2 pipeline: src=%s out=%s", args.src, args.out)
    result = run_v2_pipeline(src_dir=args.src, output_dir=args.out, params=params)
    if not result.success:
        logger.error("Run failed: %s", result.error_message)
        return 1

    logger.info(
        "Run complete — %d clusters from %d faces across %d images (noise=%d). DB: %s",
        result.n_clusters, result.n_faces, result.n_images, result.n_noise, result.db_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
