"""Standalone E2E: import from Google Photos, then run the real pipeline on it.

Proves the full round trip with NO Albumify / API / Streamlit / DB:

    Picker import  ->  local cache dir  ->  execute_spec(...)  ->  summary

The default runs a fast wiring proof (discover + geo, no ML models). Use --full
to run the production default_pipeline (face detection/clustering; needs models,
slow on first run), or --steps to pick an explicit chain.

Usage:
    .venv/Scripts/python scripts/gphotos_e2e_smoke.py
    .venv/Scripts/python scripts/gphotos_e2e_smoke.py --steps discover_images,extract_geo_metadata
    .venv/Scripts/python scripts/gphotos_e2e_smoke.py --full
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from gphotos.ingest_source import import_from_google_photos
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.run import execute_spec
from sim_bench.pipeline.spec import PipelineSpec

DEFAULT_STEPS = ["discover_images", "extract_geo_metadata"]


def _load_full_pipeline() -> list[str]:
    cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
    return list(cfg["default_pipeline"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Google Photos -> pipeline E2E smoke")
    ap.add_argument("--client-secret", default="client_secret.json")
    ap.add_argument("--out", default="./_gphotos_smoke/cache")
    ap.add_argument("--steps", help="comma-separated step names (overrides default)")
    ap.add_argument("--full", action="store_true", help="run the full default_pipeline")
    args = ap.parse_args()

    # 1) import from Google Photos
    print("[1/3] importing from Google Photos (pick photos in the browser)...")
    ingest = import_from_google_photos(args.client_secret, args.out)
    print(f"      imported {ingest.count} file(s) -> {ingest.source_directory}")
    if ingest.count == 0:
        print("      nothing imported; aborting")
        return 1

    # 2) choose + run the pipeline spec
    if args.full:
        steps = _load_full_pipeline()
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    else:
        steps = DEFAULT_STEPS
    print(f"[2/3] running pipeline: {' -> '.join(steps)}")
    ctx = PipelineContext(source_directory=ingest.source_directory)
    result = execute_spec(PipelineSpec(steps=steps), ctx, fail_fast=True)

    # 3) report whatever the run produced
    print(f"[3/3] pipeline success={result.success}")
    print(f"      images discovered: {len(getattr(ctx, 'image_paths', []) or [])}")
    geo = getattr(ctx, "geo_metadata", {}) or {}
    if geo:
        with_gps = sum(1 for m in geo.values() if getattr(m, "has_geo", False))
        print(f"      geo metadata:      {len(geo)} ({with_gps} with GPS)")
    faces = getattr(ctx, "face_records", []) or []
    if faces:
        print(f"      faces detected:    {len(faces)}")
    people = getattr(ctx, "people_clusters", {}) or {}
    if people:
        print(f"      identity clusters: {len(people)}")
    selected = getattr(ctx, "selected_images", []) or []
    if selected:
        print(f"      selected images:   {len(selected)}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
