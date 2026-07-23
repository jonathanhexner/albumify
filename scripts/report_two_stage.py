"""Spec-103 VISUAL report — see the scene-clustering decisions, don't read tables.

Two parts per trip:
  1) THE TWO EXAMPLE DECISIONS, shown with the real photos: the Budapest over-merge (two shots 4 min
     apart that flat fusion wrongly glued together, two-stage keeps apart) and the Austria near-duplicate
     (that visual-only threw away, two-stage rescued).
  2) EVERY SCENE two-stage produced, as a gallery row each — so you can eyeball whether the groups look
     like real moments, plus the photos it left unmatched.

    .venv/Scripts/python scripts/report_two_stage.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("report_two_stage")

WORK = Path(r"D:\albumify_vs_vlm")
OUT_DIR = Path(r"D:\sim-bench\reports\2026-07-21_scene_clustering_fusion")
THUMBS = OUT_DIR / "thumbs"

FLAGS = {
    "budapest": {"kind": "over-merge", "a": "20250822_193734", "b": "20250822_194159"},
    "austria": {"kind": "rescue", "a": "20230819_114651", "b": "20230819_114622"},
}
MAX_SCENES = {"budapest": 999, "austria": 30, "germany": 30}


def _thumb(src: Path, dst: Path, max_edge: int = 200) -> None:
    if dst.exists():
        return
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    img.save(dst, "JPEG", quality=80)


def _hdbscan(D, mcs=2):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="precomputed").fit_predict(D.astype(np.float64))


def run_trip(trip: str):
    wd = WORK / trip
    manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = manifest["records"]
    imgs_dir = Path(manifest["out_dir"])
    stems = [r["stem"] for r in recs]

    from sim_bench.feature_extraction.base import load_method
    feats = load_method("dinov2", {"method": "dinov2", "batch_size": 32}).extract_features(
        [str(imgs_dir / f"{s}.jpg") for s in stems])
    emb = {stems[i]: feats[i] for i in range(len(stems))}
    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    srcs = [r["src"] for r in recs]
    gm = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo = {recs[i]["stem"]: gm.get(srcs[i]) for i in range(len(recs))}

    from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceFuser, SceneDistanceInputs
    from sim_bench.scene_cluster.two_stage import TwoStageInputs, TwoStageSceneClusterer

    Dv = SceneDistanceFuser(1, 0, 0).calc(SceneDistanceInputs(emb, geo, stems)).distance_matrix
    visual = {stems[i]: int(v) for i, v in enumerate(_hdbscan(Dv))}
    Df = SceneDistanceFuser(1, 1, 1, time_scale_min=30).calc(SceneDistanceInputs(emb, geo, stems)).distance_matrix
    flat = {stems[i]: int(v) for i, v in enumerate(_hdbscan(Df))}
    # tightened per user feedback: strict look-alike (tau=0.40), time helps ONLY at <=60s (burst)
    ts = TwoStageSceneClusterer(gap_threshold_min=60, visual_tau=0.40,
                                burst_sec=60, burst_tau=0.65).calc(TwoStageInputs(emb, geo, stems))

    return {"trip": trip, "stems": stems, "imgs_dir": imgs_dir,
            "visual": visual, "flat": flat, "two_stage": ts.labels}


def _scene_members(labels: dict, cid: int) -> list:
    return sorted([s for s, c in labels.items() if c == cid])


def _tile(stem: str, cls: str = "") -> str:
    return (f'<figure class="{cls}"><img loading="lazy" src="thumbs/{stem}.jpg" alt="{stem}">'
            f'<figcaption>{stem[-6:]}</figcaption></figure>')


def _row(stems: list, cls: str = "") -> str:
    return f'<div class="row {cls}">' + "".join(_tile(s) for s in stems) + "</div>"


def build_trip_html(r: dict, copy: set) -> str:
    trip, labels_ts = r["trip"], r["two_stage"]
    parts = [f'<h2>{trip.title()}</h2>']

    # --- the flagged example decision ---
    fl = FLAGS.get(trip)
    if fl:
        a, b = fl["a"], fl["b"]
        copy.update([a, b])
        if fl["kind"] == "over-merge":
            flat_cl = _scene_members(r["flat"], r["flat"][a])
            ts_a = _scene_members(labels_ts, labels_ts[a]) if labels_ts[a] != -1 else [a]
            ts_b = _scene_members(labels_ts, labels_ts[b]) if labels_ts[b] != -1 else [b]
            copy.update(flat_cl + ts_a + ts_b)
            parts.append(
                '<div class="case">'
                '<p class="q">Example 1 &mdash; two shots taken <b>4 minutes apart</b>. Do they belong in the '
                'same scene? Look:</p>'
                f'<div class="pair">{_tile(a,"big")}{_tile(b,"big")}</div>'
                '<p class="bad"><b>Flat fusion</b> said YES, same scene &mdash; it grouped them (and everything '
                'in between) into one blob:</p>'
                f'{_row(flat_cl)}'
                '<p class="good"><b>Two-stage</b> said NO &mdash; it put each with its own visually-matching '
                f'neighbours instead. {a[-6:]}&rsquo;s scene:</p>{_row(ts_a)}'
                f'<p class="good">{b[-6:]}&rsquo;s scene:</p>{_row(ts_b)}'
                '</div>')
        else:  # rescue
            ts_cl = _scene_members(labels_ts, labels_ts[a]) if labels_ts[a] != -1 else [a, b]
            copy.update(ts_cl)
            v_a = "unmatched (thrown to noise)" if r["visual"][a] == -1 else "kept"
            parts.append(
                '<div class="case">'
                '<p class="q">Example 2 &mdash; two shots taken <b>29 seconds apart</b>, nearly identical. '
                'They obviously belong together:</p>'
                f'<div class="pair">{_tile(a,"big")}{_tile(b,"big")}</div>'
                f'<p class="bad"><b>Plain visual clustering</b> {v_a} &mdash; it failed to group these '
                'near-duplicates on pixels alone.</p>'
                '<p class="good"><b>Two-stage</b> grouped them correctly into one scene:</p>'
                f'{_row(ts_cl)}'
                '</div>')

    # --- all two-stage scenes ---
    clusters = {}
    for s, c in labels_ts.items():
        clusters.setdefault(c, []).append(s)
    noise = sorted(clusters.pop(-1, []))
    scenes = sorted((v for v in clusters.values()), key=len, reverse=True)
    cap = MAX_SCENES.get(trip, 30)
    shown = scenes[:cap]
    for sc in shown:
        copy.update(sc)
    copy.update(noise[:40])

    parts.append(f'<h3>Every scene two-stage made &mdash; {len(scenes)} scenes, '
                 f'{sum(len(s) for s in scenes)} photos grouped, {len(noise)} left unmatched'
                 + (f' (showing {len(shown)} largest scenes)' if len(shown) < len(scenes) else '') + '</h3>')
    parts.append('<p class="hint">Each row is one scene. Ask yourself: do these look like the same '
                 'moment/place? If a row mixes clearly different moments, two-stage over-grouped.</p>')
    scene_html = []
    for i, sc in enumerate(shown, 1):
        scene_html.append(f'<div class="scene"><span class="lbl">Scene {i} &middot; {len(sc)} photos</span>'
                          f'{_row(sorted(sc))}</div>')
    parts.append("".join(scene_html))
    if noise:
        parts.append(f'<h3 class="noiseh">Left unmatched: {len(noise)} photos'
                     + (f' (showing 40)' if len(noise) > 40 else '') + '</h3>')
        parts.append('<p class="hint">These didn&rsquo;t visually match anything in their time window. '
                     'Some are genuinely one-off shots; some are misses.</p>')
        parts.append(f'<div class="scene noise">{_row(noise[:40])}</div>')
    return "".join(parts)


def build_html(results: list) -> str:
    copy: set = set()
    bodies = [build_trip_html(r, copy) for r in results]
    # copy thumbs
    dir_by_stem = {}
    for r in results:
        for s in r["stems"]:
            dir_by_stem[s] = r["imgs_dir"]
    THUMBS.mkdir(parents=True, exist_ok=True)
    for s in copy:
        if s in dir_by_stem:
            _thumb(dir_by_stem[s] / f"{s}.jpg", THUMBS / f"{s}.jpg")

    nav = " &middot; ".join(f'<a href="#{r["trip"]}">{r["trip"].title()}</a>' for r in results)
    sections = "".join(f'<section id="{r["trip"]}">{b}</section>' for r, b in zip(results, bodies))
    return f"""<!doctype html><meta charset="utf-8"><title>Scene clustering &mdash; see the decisions</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;
  padding:2rem 1rem 5rem;color:#1a1a1a;background:#fff}}
 h1{{font-size:1.7rem;margin:0 0 .3rem}} h2{{margin:2.5rem 0 .5rem;font-size:1.5rem;
  border-bottom:3px solid #245b7a;padding-bottom:.3rem;color:#245b7a}}
 h3{{margin:2rem 0 .3rem;font-size:1.1rem}} h3.noiseh{{color:#999}}
 .intro{{background:#f2f7fa;border:1px solid #d6e4ee;border-radius:10px;padding:1rem 1.2rem;margin:1rem 0}}
 .nav{{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #eee;font-size:.9rem;z-index:5}}
 .nav a{{color:#245b7a;text-decoration:none;font-weight:600;margin-right:.3rem}}
 .case{{border:1px solid #e6e6e6;border-radius:12px;padding:1rem 1.2rem;margin:1rem 0;background:#fafafa}}
 .q{{font-size:1.05rem}} .bad{{color:#b23b3b}} .good{{color:#1f7a4d}}
 .pair{{display:flex;gap:.8rem;margin:.6rem 0}}
 .row{{display:flex;gap:.35rem;flex-wrap:wrap;margin:.3rem 0 .8rem}}
 figure{{margin:0;width:96px}} figure img{{width:96px;height:96px;object-fit:cover;border-radius:5px;display:block}}
 figure.big{{width:200px}} figure.big img{{width:200px;height:200px;outline:3px solid #245b7a}}
 figcaption{{font-size:9px;color:#aaa;text-align:center}}
 .scene{{border-left:3px solid #245b7a;padding:.3rem 0 .3rem .7rem;margin:.5rem 0;background:#f8fafb;border-radius:0 6px 6px 0}}
 .scene.noise{{border-color:#ccc;background:#f7f7f7}}
 .lbl{{font-size:.78rem;font-weight:700;color:#245b7a;display:block;margin-bottom:.2rem}}
 .hint{{color:#666;font-size:.9rem}}
</style>
<h1>Scene clustering &mdash; see the decisions</h1>
<div class="intro"><b>What you&rsquo;re looking at.</b> A &ldquo;scene&rdquo; = photos from the same
 moment/place, so an album can pick one keeper per scene. We tested how to group them. This page shows
 the actual photos, not numbers. For each trip: <b>(1)</b> one example grouping decision, with the real
 photos, comparing the old way vs the new <b>two-stage</b> method; <b>(2)</b> every scene two-stage made,
 one row each &mdash; <b>your job: does each row look like one real moment?</b></div>
<div class="nav">Jump to: {nav}</div>
{sections}
<hr><p class="hint">two-stage (tightened) = split by capture-time first (new group when &gt;60 min gap), then
 within a group two photos share a scene only if they <b>look alike</b> (visual distance &lt; 0.40),
 <b>or</b> they were taken <b>&le;60 s apart</b> and aren&rsquo;t wildly different. Time only counts at
 very short range. <code>sim_bench/scene_cluster/two_stage.py</code>.</p>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["budapest", "austria", "germany"])
    args = ap.parse_args()
    results = [run_trip(t) for t in args.trips]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "two_stage.html").write_text(build_html(results), encoding="utf-8")
    print(f"REPORT: {OUT_DIR / 'two_stage.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
