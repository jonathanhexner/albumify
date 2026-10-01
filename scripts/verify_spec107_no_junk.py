#!/usr/bin/env python
"""spec-107 P2 verification: with triage ON, confirm 0 junk reaches the curated album.

Same 75-image Slovakia subset as the 2026-10-01 analysis (screenshots included), but now the
default pipeline runs triage_metadata + triage_content first. Assert: screenshots/docs land in
context.set_aside and NONE appear in the curated album (was 6/20 before triage).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
for _n in ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]:
    if not hasattr(_mu, _n) and hasattr(_pu, _n):
        setattr(_mu, _n, getattr(_pu, _n))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = Path("D:/Slovakia26")
WORK = ROOT.parent / "_slovakia_triage_work"
SUBSET = 75


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    all_imgs = sorted(SRC.glob("*.jpg"))
    shots = [p for p in all_imgs if p.name.lower().startswith("screenshot")]
    photos = [p for p in all_imgs if not p.name.lower().startswith("screenshot")]
    idx = np.linspace(0, len(photos) - 1, SUBSET - len(shots)).round().astype(int)
    subset = shots + [photos[i] for i in dict.fromkeys(idx)]
    for p in subset:
        src = Image.open(p)
        exif = src.info.get("exif")  # preserve camera EXIF so the metadata tier is tested honestly
        im = src.convert("RGB"); im.thumbnail((768, 768))
        im.save(WORK / p.name, quality=90, **({"exif": exif} if exif else {}))
    print(f"subset {len(subset)} ({len(shots)} screenshots) -> {WORK}")

    import yaml
    from sim_bench.albumify_vs_vlm.albumify_arm import _resolve_steps, PIPELINE_YAML
    from sim_bench.albumify_vs_vlm.curation import select_top_k_coverage_first, chronological_order
    from sim_bench.pipeline.run import execute_spec
    from sim_bench.pipeline.spec import PipelineSpec
    from sim_bench.pipeline.context import PipelineContext

    doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    steps = _resolve_steps(doc, "faces")
    step_configs = {name: (doc.get(name) or {}) for name in steps}
    step_configs["select_best"] = {**(step_configs.get("select_best") or {}), "max_images_per_cluster": 2}
    ctx = PipelineContext(source_directory=WORK)
    print(f"running faces pipeline WITH triage ({len(steps)} steps) ...")
    res = execute_spec(PipelineSpec(steps=steps, step_configs=step_configs), ctx, fail_fast=True)
    assert res.success, f"pipeline failed: {res.failed_step} {res.error_message}"

    scores = {Path(p).stem: float(s) for p, s in (ctx.composite_scores or {}).items()}
    labels = {Path(p).stem: int(c) for p, c in (ctx.scene_cluster_labels or {}).items()}
    selected = [Path(p).stem for p in (ctx.selected_images or [])]
    chosen = select_top_k_coverage_first(selected, scores, labels, 20)
    if len(chosen) < 20:
        rest = [s for s in scores if s not in set(chosen)]
        chosen += select_top_k_coverage_first(rest, scores, labels, 20 - len(chosen))
    album = chronological_order(chosen)

    set_aside = {Path(e["image"]).name for e in ctx.set_aside}
    review = {Path(e["image"]).name for e in ctx.triage_review}
    album_names = {s + ".jpg" for s in album}
    junk_in_album = [n for n in album_names if n.lower().startswith("screenshot")]

    print("\n=== spec-107 P2 VERIFICATION ===")
    print(f"set_aside ({len(set_aside)}): by tier -> "
          f"metadata={sum(1 for e in ctx.set_aside if e['tier']=='metadata')}, "
          f"content={sum(1 for e in ctx.set_aside if e['tier']=='content')}")
    print(f"triage_review (kept, flagged): {len(review)}")
    print(f"scored images (survivors): {len(scores)}")
    print(f"album size: {len(album)}")
    print(f"screenshots in album: {len(junk_in_album)} {junk_in_album}")
    print(f"any set_aside image in album: {sorted(album_names & set_aside)}")
    ok = len(junk_in_album) == 0 and not (album_names & set_aside)
    print(f"\nRESULT: {'PASS - 0 junk in album (was 6/20 before triage)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
