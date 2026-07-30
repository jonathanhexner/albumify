"""Spec-103 PATH A — YOUR production clusterer (HDBSCAN) + a short-range time boost on the distance.

No new clustering method: this reuses the SAME HDBSCAN settings as sim_bench/pipeline/steps/cluster_scenes.py
(min_cluster_size=2, min_samples=2, cosine, eom). The only change is the distance it clusters:

    d(i,j) = visual_cosine_distance(i,j) * discount(dt)
    discount(dt) = 1 - boost * exp(-dt_seconds / tau_sec)      # boost=0.6, tau=60s

So two photos taken <=~1 min apart get their distance shrunk (a strong same-moment pull); photos minutes
apart get discount ~= 1 (time does NOTHING, looks decide). Time only ever PULLS, never pushes apart -- the
fix for the old additive blend's over-merge. HDBSCAN's density extraction resists the single-linkage
chaining that produced the 49-photo mega-scene.

Renders the same visual report: baseline (your clusterer today) vs Path A, the two flagged cases with real
photos, every Path A scene as a gallery row, and a chaining check (largest-scene visual diameter).

    .venv/Scripts/python scripts/report_path_a.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("report_path_a")

WORK = Path(r"D:\albumify_vs_vlm")
OUT_DIR = Path(r"D:\sim-bench\reports\2026-07-21_scene_clustering_fusion")
THUMBS = OUT_DIR / "thumbs"

FLAGS = {
    "budapest": {"kind": "over-merge", "a": "20250822_193734", "b": "20250822_194159"},
    "austria": {"kind": "rescue", "a": "20230819_114651", "b": "20230819_114622"},
}
MAX_SCENES = {"budapest": 999, "austria": 30, "germany": 30}
BOOST = 0.6
TAU_SEC = 60.0


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


def _hdbscan_production(D):
    """Your cluster_scenes settings, on a precomputed distance."""
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed",
                           cluster_selection_method="eom").fit_predict(D.astype(np.float64))


def run_trip(trip: str):
    wd = WORK / trip
    manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = manifest["records"]
    imgs_dir = Path(manifest["out_dir"])
    stems = [r["stem"] for r in recs]

    from sim_bench.feature_extraction.base import load_method
    feats = load_method("dinov2", {"method": "dinov2", "batch_size": 32}).extract_features(
        [str(imgs_dir / f"{s}.jpg") for s in stems])
    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    from sim_bench.scene_cluster.geo_time_fusion import _time_from
    srcs = [r["src"] for r in recs]
    gm = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo = {stems[i]: gm.get(srcs[i]) for i in range(len(recs))}
    times = [_time_from(stems[i], geo[stems[i]]) for i in range(len(stems))]

    E = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    visual_d = 1.0 - np.clip(E @ E.T, -1.0, 1.0)

    # Path A distance: visual * short-range time discount (one-sided: only shrinks, never grows)
    n = len(stems)
    D = visual_d.copy()
    for a in range(n):
        for b in range(a + 1, n):
            if times[a] and times[b]:
                dt = abs((times[a] - times[b]).total_seconds())
                disc = 1.0 - BOOST * math.exp(-dt / TAU_SEC)
                D[a, b] = D[b, a] = visual_d[a, b] * disc
    np.fill_diagonal(D, 0.0)

    baseline = {stems[i]: int(v) for i, v in enumerate(_hdbscan_production(visual_d))}
    path_a = {stems[i]: int(v) for i, v in enumerate(_hdbscan_production(D))}
    return {"trip": trip, "stems": stems, "imgs_dir": imgs_dir, "visual_d": visual_d,
            "baseline": baseline, "path_a": path_a}


def _members(labels, cid):
    return sorted([s for s, c in labels.items() if c == cid])


def _diameter(scene, stems, visual_d):
    idx = {s: i for i, s in enumerate(stems)}
    ii = [idx[s] for s in scene]
    sub = visual_d[np.ix_(ii, ii)]
    return float(sub.max())


def _tile(stem, cls=""):
    return (f'<figure class="{cls}"><img loading="lazy" src="thumbs/{stem}.jpg" alt="{stem}">'
            f'<figcaption>{stem[-6:]}</figcaption></figure>')


def _row(stems, cls=""):
    return f'<div class="row {cls}">' + "".join(_tile(s) for s in stems) + "</div>"


def build_trip_html(r, copy):
    trip, la = r["trip"], r["path_a"]
    parts = [f'<h2>{trip.title()}</h2>']
    fl = FLAGS.get(trip)
    if fl:
        a, b = fl["a"], fl["b"]
        copy.update([a, b])
        base_a = _members(r["baseline"], r["baseline"][a]) if r["baseline"][a] != -1 else [a]
        pa_a = _members(la, la[a]) if la[a] != -1 else [a]
        pa_b = _members(la, la[b]) if la[b] != -1 else [b]
        copy.update(base_a + pa_a + pa_b)
        if fl["kind"] == "over-merge":
            same = "the SAME scene" if la[a] == la[b] and la[a] != -1 else "SEPARATE scenes"
            parts.append(
                '<div class="case"><p class="q">Example 1 &mdash; two shots <b>4 minutes apart</b>. '
                'Different places?</p>'
                f'<div class="pair">{_tile(a,"big")}{_tile(b,"big")}</div>'
                f'<p class="good"><b>Path A</b> put them in <b>{same}</b>. '
                f'{a[-6:]}&rsquo;s scene:</p>{_row(pa_a)}'
                f'<p class="good">{b[-6:]}&rsquo;s scene:</p>{_row(pa_b)}</div>')
        else:
            joined = (la[a] == la[b] and la[a] != -1)
            v_base = "threw one to noise" if r["baseline"][a] == -1 or r["baseline"][b] == -1 else "kept both"
            parts.append(
                '<div class="case"><p class="q">Example 2 &mdash; two shots <b>29 seconds apart</b>, '
                'nearly identical.</p>'
                f'<div class="pair">{_tile(a,"big")}{_tile(b,"big")}</div>'
                f'<p class="bad">Your clusterer <b>today</b> (visual only) {v_base}.</p>'
                f'<p class="good"><b>Path A</b> {"grouped them" if joined else "did NOT group them"} '
                f'(short-range time boost). The scene:</p>{_row(pa_a if joined else [a,b])}</div>')

    clusters = {}
    for s, c in la.items():
        clusters.setdefault(c, []).append(s)
    noise = sorted(clusters.pop(-1, []))
    scenes = sorted(clusters.values(), key=len, reverse=True)
    cap = MAX_SCENES.get(trip, 30)
    shown = scenes[:cap]
    for sc in shown:
        copy.update(sc)
    copy.update(noise[:40])

    diam = _diameter(scenes[0], r["stems"], r["visual_d"]) if scenes else 0.0
    parts.append(
        f'<h3>Every Path A scene &mdash; {len(scenes)} scenes, '
        f'{sum(len(s) for s in scenes)} grouped, {len(noise)} unmatched'
        + (f' (largest {len(shown)} shown)' if len(shown) < len(scenes) else '') + '</h3>')
    parts.append(f'<p class="hint">Largest scene = <b>{len(scenes[0]) if scenes else 0} photos</b>, '
                 f'visual diameter <b>{diam:.2f}</b> '
                 f'({"NO chaining — tight" if diam < 0.75 else "still some spread"}; '
                 f'compare the old two-stage&rsquo;s 49 photos / 1.01 diameter).</p>')
    for i, sc in enumerate(shown, 1):
        parts.append(f'<div class="scene"><span class="lbl">Scene {i} &middot; {len(sc)} photos</span>'
                     f'{_row(sorted(sc))}</div>')
    if noise:
        parts.append(f'<h3 class="noiseh">Left unmatched: {len(noise)}'
                     + (' (showing 40)' if len(noise) > 40 else '') + '</h3>')
        parts.append(f'<div class="scene noise">{_row(noise[:40])}</div>')
    return "".join(parts)


def build_html(results):
    copy = set()
    bodies = [build_trip_html(r, copy) for r in results]
    dir_by_stem = {s: r["imgs_dir"] for r in results for s in r["stems"]}
    THUMBS.mkdir(parents=True, exist_ok=True)
    for s in copy:
        if s in dir_by_stem:
            _thumb(dir_by_stem[s] / f"{s}.jpg", THUMBS / f"{s}.jpg")
    nav = " &middot; ".join(f'<a href="#{r["trip"]}">{r["trip"].title()}</a>' for r in results)
    sections = "".join(f'<section id="{r["trip"]}">{b}</section>' for r, b in zip(results, bodies))
    return f"""<!doctype html><meta charset="utf-8"><title>Path A &mdash; your HDBSCAN + time boost</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;
  padding:2rem 1rem 5rem;color:#1a1a1a;background:#fff}}
 h1{{font-size:1.7rem;margin:0 0 .3rem}} h2{{margin:2.5rem 0 .5rem;font-size:1.5rem;
  border-bottom:3px solid #7a4b24;padding-bottom:.3rem;color:#7a4b24}}
 h3{{margin:2rem 0 .3rem;font-size:1.1rem}} h3.noiseh{{color:#999}}
 .intro{{background:#faf5ef;border:1px solid #eaddcb;border-radius:10px;padding:1rem 1.2rem;margin:1rem 0}}
 .nav{{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #eee;font-size:.9rem}}
 .nav a{{color:#7a4b24;text-decoration:none;font-weight:600;margin-right:.3rem}}
 .case{{border:1px solid #e6e6e6;border-radius:12px;padding:1rem 1.2rem;margin:1rem 0;background:#fafafa}}
 .q{{font-size:1.05rem}} .bad{{color:#b23b3b}} .good{{color:#1f7a4d}}
 .pair{{display:flex;gap:.8rem;margin:.6rem 0}}
 .row{{display:flex;gap:.35rem;flex-wrap:wrap;margin:.3rem 0 .8rem}}
 figure{{margin:0;width:96px}} figure img{{width:96px;height:96px;object-fit:cover;border-radius:5px;display:block}}
 figure.big{{width:200px}} figure.big img{{width:200px;height:200px;outline:3px solid #7a4b24}}
 figcaption{{font-size:9px;color:#aaa;text-align:center}}
 .scene{{border-left:3px solid #7a4b24;padding:.3rem 0 .3rem .7rem;margin:.5rem 0;background:#fbf9f6;border-radius:0 6px 6px 0}}
 .scene.noise{{border-color:#ccc;background:#f7f7f7}}
 .lbl{{font-size:.78rem;font-weight:700;color:#7a4b24;display:block;margin-bottom:.2rem}}
 .hint{{color:#666;font-size:.9rem}}
</style>
<h1>Path A &mdash; your HDBSCAN, with a smarter distance</h1>
<div class="intro"><b>What changed vs today:</b> nothing about the clusterer &mdash; this is your exact
 <code>cluster_scenes</code> HDBSCAN (min_cluster_size=2, cosine, eom). The only change is the distance:
 two photos taken <b>&le;~1 min apart get pulled together</b>; anything more than a couple minutes apart
 is judged <b>purely on looks</b>. Time only ever helps at short range &mdash; it never forces distant
 photos apart. <b>Your job:</b> do the scene rows look tight, and is the 49-photo chaining gone?</div>
<div class="nav">Jump to: {nav}</div>
{sections}
<hr><p class="hint">Path A distance: <code>visual_cos * (1 - 0.6*exp(-seconds/60))</code>, clustered by your
 production HDBSCAN. No new clustering method. <code>scripts/report_path_a.py</code>.</p>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["budapest", "austria", "germany"])
    args = ap.parse_args()
    results = [run_trip(t) for t in args.trips]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "path_a.html").write_text(build_html(results), encoding="utf-8")
    for r in results:
        clus = {}
        for s, c in r["path_a"].items():
            clus.setdefault(c, []).append(s)
        noise = len(clus.pop(-1, []))
        big = max(clus.values(), key=len)
        print(f"{r['trip']}: {len(clus)} scenes, {noise} noise, largest={len(big)} "
              f"(diameter {_diameter(big, r['stems'], r['visual_d']):.2f})")
    print(f"REPORT: {OUT_DIR / 'path_a.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
