#!/usr/bin/env python
"""spec-109 T0.1 — full default_pipeline per-step timing baseline (CPU).

The 2026-10-01 Slovakia run used the lean `faces` variant (dropped occlusion/tilt/straighten). This
runs the REAL `default_pipeline` (38 steps incl. score_occlusion CLIP + score_tilt GeoCalib) on the
768px work set, capturing per-step `duration_ms` → honest numbers for the GPU/parallel sizing.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
for _n in ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]:
    if not hasattr(_mu, _n) and hasattr(_pu, _n):
        setattr(_mu, _n, getattr(_pu, _n))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORK = ROOT.parent / "_slovakia_triage_work"
OUT = ROOT / "reports" / "2026-10-04_spec109_baseline"


def main():
    import yaml
    from sim_bench.pipeline.run import execute_spec
    from sim_bench.pipeline.spec import PipelineSpec
    from sim_bench.pipeline.context import PipelineContext

    doc = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    steps = doc["default_pipeline"]
    step_configs = {s: (doc.get(s) or {}) for s in steps}
    ctx = PipelineContext(source_directory=WORK)

    n_in = len(list(WORK.glob("*.jpg")))
    print(f"full default_pipeline ({len(steps)} steps) on {n_in} images (CPU) ...")
    t0 = time.time()
    result = execute_spec(PipelineSpec(steps=steps, step_configs=step_configs), ctx, fail_fast=True)
    wall = time.time() - t0
    if not result.success:
        print(f"FAILED at {result.failed_step}: {result.error_message}")
        return 1

    n_scored = len(ctx.composite_scores or {}) or n_in
    rows = sorted(
        [{"step": r.step_name, "s": round(r.duration_ms / 1000, 1),
          "ms_per_img": round(r.duration_ms / max(1, n_scored), 1)} for r in result.step_results],
        key=lambda d: -d["s"])
    total_s = result.total_duration_ms / 1000

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(
        {"n_in": n_in, "n_scored": n_scored, "wall_s": round(wall, 1),
         "total_s": round(total_s, 1), "set_aside": len(ctx.set_aside or []),
         "album_theme": (ctx.album_theme or {}).get("album_type"), "steps": rows}, indent=2), encoding="utf-8")

    print(f"\n=== spec-109 baseline (full default_pipeline, CPU) ===")
    print(f"images in={n_in} scored={n_scored} set_aside={len(ctx.set_aside or [])} | "
          f"total {total_s:.0f}s = {total_s/max(1,n_scored):.1f}s/img")
    print("slowest 12 steps (s | ms/img):")
    for r in rows[:12]:
        print(f"  {r['step']:30} {r['s']:7.1f}s  {r['ms_per_img']:8.1f}")
    print(f"timing.json: {OUT/'timing.json'}")


if __name__ == "__main__":
    sys.exit(main())
