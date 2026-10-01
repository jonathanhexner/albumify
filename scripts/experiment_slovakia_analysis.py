#!/usr/bin/env python
"""
Slovakia26 analysis: (1) triage irrelevant photos, (2) run Albumify, (3) PERFORMANCE/timing, (4) data.

Collects data only (no HTML) -> reports/2026-10-01_slovakia_analysis/results.json + thumbs.
The comprehensive report is authored from that JSON.

Captures per-step pipeline timing (duration_ms) + per-phase wall time so we can answer
"what is taking so long". Inlines run_albumify_arm's logic (reusing its helpers) because the
public function discards the per-step timing we need.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
from PIL import Image

import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
for _n in ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]:
    if not hasattr(_mu, _n) and hasattr(_pu, _n):
        setattr(_mu, _n, getattr(_pu, _n))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa
from ram.models import ram_plus  # noqa
from ram import inference_ram, get_transform  # noqa

SRC = Path("D:/Slovakia26")
OUT = ROOT / "reports" / "2026-10-01_slovakia_analysis"
IMGS = OUT / "imgs"
WORK = ROOT.parent / "_slovakia_work"
CKPT = ROOT / "models" / "ram" / "ram_plus_swin_large_14m.pth"
SUBSET = 75
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
if SMOKE:
    SUBSET = 6

JUNK_TAGS = {"screenshot", "document", "text", "poster", "movie poster", "menu", "receipt",
             "website", "logo", "label", "movie ticket", "road sign", "parking sign", "comic",
             "cartoon", "diagram", "chart", "map", "flipchart", "letter logo", "comic book"}
PERSON_TAGS = {"person", "man", "woman", "child", "girl", "boy", "people", "crowd", "group",
               "baby", "family", "selfie", "portrait"}


def has_camera_exif(p):
    try:
        ex = Image.open(p).getexif()
        return bool(ex.get(271) or ex.get(272))
    except Exception:
        return False


def main():
    IMGS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    phases = {}
    all_imgs = sorted([p for p in SRC.glob("*.jpg")])
    print(f"{len(all_imgs)} jpgs in {SRC}")

    # ---- metadata triage on ALL ----
    t = time.time()
    meta = []
    for p in all_imgs:
        meta.append(dict(name=p.name, shot_name=p.name.lower().startswith("screenshot"),
                         cam_exif=has_camera_exif(p)))
    phases["metadata_triage_all"] = round(time.time() - t, 1)
    flagged = [m for m in meta if m["shot_name"] or not m["cam_exif"]]
    print(f"triage(all): shot-name={sum(m['shot_name'] for m in meta)}, "
          f"no-cam-exif={sum(not m['cam_exif'] for m in meta)}, flagged={len(flagged)} "
          f"({phases['metadata_triage_all']}s)")

    # ---- subset: all flagged + evenly-spaced real photos ----
    shot_names = {m["name"] for m in flagged}
    shots = [p for p in all_imgs if p.name in shot_names]
    photos = [p for p in all_imgs if p.name not in shot_names]
    n_photos = max(1, SUBSET - len(shots))
    idx = np.linspace(0, len(photos) - 1, min(n_photos, len(photos))).round().astype(int)
    subset = (shots[: (2 if SMOKE else len(shots))]) + [photos[i] for i in dict.fromkeys(idx)]
    print(f"subset: {len(subset)}")

    # ---- downsample subset to 768px + thumbs ----
    t = time.time()
    for p in subset:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print("skip", p.name, e); continue
        w = im.copy(); w.thumbnail((768, 768)); w.save(WORK / p.name, quality=90)
        th = im.copy(); th.thumbnail((300, 300)); th.save(IMGS / p.name, quality=82)
    phases["downsample_subset"] = round(time.time() - t, 1)

    # ---- RAM++ content tags on subset ----
    t = time.time()
    print("loading RAM++ ...")
    ram = ram_plus(pretrained=str(CKPT), image_size=384, vit="swin_l")
    ram.eval().to("cpu")
    tf = get_transform(image_size=384)
    phases["ram_load"] = round(time.time() - t, 1)
    t = time.time()
    recs = {}
    for k, p in enumerate(subset, 1):
        wp = WORK / p.name
        if not wp.exists():
            continue
        try:
            with torch.no_grad():
                res = inference_ram(tf(Image.open(wp).convert("RGB")).unsqueeze(0), ram)
            tags = [x.strip() for x in str(res[0] if isinstance(res, (list, tuple)) else res).split("|") if x.strip()]
        except Exception as e:
            print("ram skip", p.name, e); tags = []
        low = set(x.lower() for x in tags)
        recs[p.name] = dict(name=p.name, flagged_meta=p.name in shot_names, ram_tags=tags[:14],
                            junk_tags=sorted(low & JUNK_TAGS), has_person=bool(low & PERSON_TAGS))
        if k % 15 == 0 or SMOKE:
            print(f"  ram [{k}/{len(subset)}]")
    phases["ram_inference"] = round(time.time() - t, 1)
    n_ram = len([r for r in recs.values()])
    print(f"RAM++: load {phases['ram_load']}s, inference {phases['ram_inference']}s "
          f"({phases['ram_inference']/max(1,n_ram):.1f}s/img)")
    del ram

    # ---- Albumify (inline run_albumify_arm, capturing per-step timing) ----
    album = {"ok": False}
    timing = {"phases": phases}
    try:
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
        spec = PipelineSpec(steps=steps, step_configs=step_configs)
        ctx = PipelineContext(source_directory=WORK)

        print(f"running Albumify (faces pipeline, {len(steps)} steps) ...")
        t = time.time()
        result = execute_spec(spec, ctx, fail_fast=True)
        album_wall = round(time.time() - t, 1)
        if not result.success:
            raise RuntimeError(f"failed at {result.failed_step}: {result.error_message}")

        sf = getattr(ctx, "straightened_from", {}) or {}
        stem = lambda p: Path(sf.get(p, p)).stem
        scores = {stem(p): float(s) for p, s in (ctx.composite_scores or {}).items()}
        labels = {stem(p): int(c) for p, c in (ctx.scene_cluster_labels or {}).items()}
        clusters = {}
        for p, c in labels.items():
            clusters.setdefault(c, []).append(p)
        cl_size = {c: len(v) for c, v in clusters.items()}
        selected = [stem(p) for p in (ctx.selected_images or [])]
        chosen = select_top_k_coverage_first(selected, scores, labels, 20)
        if len(chosen) < 20:
            rest = [s for s in scores if s not in set(chosen)]
            chosen += select_top_k_coverage_first(rest, scores, labels, 20 - len(chosen))
        order = chronological_order(chosen)

        n_scored = len(scores)
        step_ms = [{"step": sr.step_name, "ms": sr.duration_ms,
                    "ms_per_img": round(sr.duration_ms / max(1, n_scored), 1)}
                   for sr in result.step_results]
        step_ms.sort(key=lambda d: -d["ms"])
        timing.update(album_total_ms=result.total_duration_ms, album_wall_s=album_wall,
                      n_scored=n_scored, steps=step_ms)
        album = dict(ok=True, k=20, order=order, n_scene_clusters=len(clusters),
                     picks=[{"id": s, "score": round(scores.get(s, 0.0), 4),
                             "cluster": labels.get(s, -1)} for s in order],
                     scores=scores, labels=labels, cl_size=cl_size)
        print(f"album: {len(order)} picks, {len(clusters)} clusters, "
              f"{result.total_duration_ms/1000:.0f}s total ({result.total_duration_ms/max(1,n_scored):.0f} ms/img)")
        print("slowest steps:", ", ".join(f"{s['step']} {s['ms']/1000:.0f}s" for s in step_ms[:5]))
    except Exception as e:
        import traceback; traceback.print_exc()
        album = {"ok": False, "error": str(e)}

    # ---- merge, behavioral, leak ----
    album_stems = set(album.get("order", []))
    junk_in_album, singletons, util = [], [], []
    for name, r in recs.items():
        st = Path(name).stem
        r["in_album"] = st in album_stems
        r["composite"] = album.get("scores", {}).get(st)
        cid = album.get("labels", {}).get(st)
        r["scene_cluster"] = cid
        r["singleton"] = cid is not None and album.get("cl_size", {}).get(cid, 2) == 1
        is_junk = r["flagged_meta"] or bool(r["junk_tags"])
        if r["in_album"] and is_junk:
            junk_in_album.append(name)
        if r["singleton"]:
            singletons.append(name)
        if r["singleton"] and not r["has_person"] and not is_junk:
            util.append(name)

    def mean_score(names):
        vs = [recs[n]["composite"] for n in names if recs[n].get("composite") is not None]
        return round(float(np.mean(vs)), 3) if vs else None
    junk_names = [n for n, r in recs.items() if r["flagged_meta"] or r["junk_tags"]]
    real_names = [n for n in recs if n not in set(junk_names)]

    results = dict(
        folder=str(SRC), total_images=len(all_imgs),
        triage_all=dict(screenshots_by_name=sum(m["shot_name"] for m in meta),
                        no_camera_exif=sum(not m["cam_exif"] for m in meta),
                        total_flagged=len(flagged), flagged_names=[m["name"] for m in flagged][:40]),
        subset_n=len(subset),
        album={k: album[k] for k in ("ok", "k", "order", "n_scene_clusters", "picks")} if album.get("ok") else album,
        timing=timing,
        images=list(recs.values()),
        behavioral=dict(singletons=singletons, utilitarian_candidates=util),
        leak=dict(junk_in_album=junk_in_album, junk_mean_score=mean_score(junk_names),
                  real_mean_score=mean_score(real_names)),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"roll {len(all_imgs)} | flagged-meta {len(flagged)} | subset {len(subset)} | "
          f"ram junk-content {len([r for r in recs.values() if r['junk_tags']])}")
    if album.get("ok"):
        print(f"album {album['k']} picks | junk leaked {len(junk_in_album)}: {junk_in_album}")
        print(f"mean composite junk {results['leak']['junk_mean_score']} vs real {results['leak']['real_mean_score']}")
        print(f"timing phases: {phases}")
    print(f"singletons {len(singletons)} | utilitarian {len(util)}: {util[:10]}")
    print(f"results: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
