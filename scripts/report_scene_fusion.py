"""Spec-103 report — before/after scene-fusion galleries per trip.

Re-runs the three clustering variants (visual / +time / +time+geo) from exp_scene_fusion, then
emits an inspectable HTML report: summary stats + the photos RESCUED FROM THE NOISE BUCKET (orphans
in visual-only that joined a real scene once time/geo was fused in), shown next to the scene they
joined. Downscaled thumbnails are copied into the report folder (experiment-report mandate).

    .venv/Scripts/python scripts/report_scene_fusion.py --trips budapest austria germany
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("report_scene_fusion")

WORK = Path(r"D:\albumify_vs_vlm")
REPORT_ROOT = Path(r"D:\sim-bench\reports")


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


def _cluster(D: np.ndarray, mcs: int = 2):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="precomputed").fit_predict(D.astype(np.float64))


def _labels_map(labels, ids) -> dict:
    return {ids[i]: int(labels[i]) for i in range(len(ids))}


def run_trip(trip: str, mcs: int, thumbs_dir: Path) -> dict:
    wd = WORK / trip
    manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = manifest["records"]
    imgs_dir = Path(manifest["out_dir"])
    stems = [r["stem"] for r in recs]
    ws_paths = [str(imgs_dir / f"{s}.jpg") for s in stems]

    logger.warning("[%s] DINOv2 embeddings for %d images ...", trip, len(stems))
    from sim_bench.feature_extraction.base import load_method
    feats = load_method("dinov2", {"method": "dinov2", "batch_size": 32}).extract_features(ws_paths)
    emb = {stems[i]: feats[i] for i in range(len(stems))}

    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    srcs = [r["src"] for r in recs]
    gmeta_by_src = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo_meta = {recs[i]["stem"]: gmeta_by_src.get(srcs[i]) for i in range(len(recs))}
    n_gps = sum(1 for m in geo_meta.values() if m is not None and m.lat is not None)

    from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceFuser, SceneDistanceInputs

    def dist(w_time, w_geo):
        f = SceneDistanceFuser(w_visual=1.0, w_time=w_time, w_geo=w_geo)
        return f.calc(SceneDistanceInputs(emb, geo_meta, stems)).distance_matrix

    variants = {"visual": dist(0.0, 0.0), "time": dist(1.0, 0.0), "fused": dist(1.0, 1.0)}
    lab = {k: _labels_map(_cluster(D, mcs), stems) for k, D in variants.items()}

    def stats(lm):
        clusters = {}
        for s, c in lm.items():
            clusters.setdefault(c, []).append(s)
        noise = clusters.pop(-1, [])
        return len(clusters), len(noise), clusters

    nc_v, nn_v, cl_v = stats(lab["visual"])
    nc_t, nn_t, _ = stats(lab["time"])
    nc_f, nn_f, cl_f = stats(lab["fused"])

    # Rescued: orphan (-1) in visual-only, but in a real cluster in fused.
    rescued = [s for s in stems if lab["visual"][s] == -1 and lab["fused"][s] != -1]
    # Group each rescued photo by the fused cluster it joined; show ALL other members of that scene,
    # flagging which were themselves orphans in visual-only (a scene built entirely from ex-orphans is
    # a genuinely NEW group; one with established members means the orphan joined an existing scene).
    rescue_groups = []
    for s in rescued:
        fc = lab["fused"][s]
        others = [m for m in cl_f[fc] if m != s]
        members = [{"stem": m, "orphan": lab["visual"][m] == -1} for m in others[:6]]
        n_established = sum(1 for m in others if lab["visual"][m] != -1)
        rescue_groups.append({"photo": s, "joined_cluster": fc, "scene_size": len(others) + 1,
                              "n_established": n_established, "members": members})

    # copy thumbs for everything shown
    shown = set(rescued)
    for g in rescue_groups:
        shown.update(m["stem"] for m in g["members"])
    for s in shown:
        _thumb(imgs_dir / f"{s}.jpg", thumbs_dir / f"{s}.jpg")

    return {
        "trip": trip, "n": len(stems), "n_gps": n_gps,
        "visual": (nc_v, nn_v), "time": (nc_t, nn_t), "fused": (nc_f, nn_f),
        "rescued": rescued, "rescue_groups": rescue_groups,
    }


def _tile(stem: str, cls: str = "") -> str:
    return (f'<figure class="{cls}"><img loading="lazy" src="thumbs/{stem}.jpg" alt="{stem}">'
            f'<figcaption>{stem}</figcaption></figure>')


def build_html(results: list[dict]) -> str:
    rows = []
    for r in results:
        gps_pct = round(100 * r["n_gps"] / r["n"])
        rows.append(
            f'<tr><td><b>{r["trip"]}</b></td><td>{r["n"]}</td><td>{gps_pct}%</td>'
            f'<td>{r["visual"][0]} / <b>{r["visual"][1]}</b></td>'
            f'<td>{r["time"][0]} / <b>{r["time"][1]}</b></td>'
            f'<td>{r["fused"][0]} / <b>{r["fused"][1]}</b></td>'
            f'<td>&minus;{r["visual"][1] - r["fused"][1]} '
            f'(&minus;{round(100 * (r["visual"][1] - r["fused"][1]) / max(1, r["visual"][1]))}%)</td></tr>'
        )
    summary = (
        '<table><tr><th>Trip</th><th>Photos</th><th>GPS</th>'
        '<th>Visual only<br>clusters / <b>noise</b></th>'
        '<th>+ Time<br>clusters / <b>noise</b></th>'
        '<th>+ Time + Geo<br>clusters / <b>noise</b></th>'
        '<th>Orphans rescued</th></tr>' + "".join(rows) + '</table>'
    )

    galleries = []
    for r in results:
        cards = []
        for g in r["rescue_groups"][:24]:  # cap per trip
            nb = "".join(_tile(m["stem"], "orphan" if m["orphan"] else "") for m in g["members"])
            if g["n_established"] == 0:
                verdict = f'formed a NEW {g["scene_size"]}-photo scene (all were orphans) &rarr;'
            else:
                verdict = f'joined a {g["scene_size"]}-photo scene ({g["n_established"]} already clustered) &rarr;'
            cards.append(
                f'<div class="card">{_tile(g["photo"], "hero")}'
                f'<span class="arrow">{verdict}</span><div class="nbs">{nb}</div></div>'
            )
        shown = len(r["rescue_groups"][:24])
        total = len(r["rescued"])
        more = f' (showing {shown} of {total})' if total > shown else ''
        galleries.append(
            f'<h2>{r["trip"].title()} — {total} orphans rescued from the noise bucket{more}</h2>'
            f'<p class="hint">Left (red outline) = a photo visual-only clustering dumped as an orphan. '
            f'Right = the rest of the scene it joined once time/geo was fused in '
            f'(<span style="outline:2px dashed #e08e2b;padding:0 2px">dashed</span> = also an ex-orphan; '
            f'plain = already clustered on pixels alone).</p>'
            f'<div class="cards">{"".join(cards)}</div>'
        )

    return f"""<!doctype html><meta charset="utf-8"><title>Scene fusion — before/after</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.6rem}} h2{{margin-top:2.5rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:14px}}
 th,td{{border:1px solid #ddd;padding:.5rem .6rem;text-align:center}} th{{background:#f6f6f6}}
 td:first-child,th:first-child{{text-align:left}}
 .hint{{color:#666;font-size:13px}}
 .cards{{display:flex;flex-direction:column;gap:.6rem}}
 .card{{display:flex;align-items:center;gap:.8rem;border:1px solid #eee;border-radius:8px;padding:.5rem;background:#fafafa}}
 .arrow{{color:#c0392b;font-weight:600;white-space:nowrap;font-size:13px}}
 .nbs{{display:flex;gap:.4rem;flex-wrap:wrap}}
 figure{{margin:0;width:110px}} figure img{{width:110px;height:110px;object-fit:cover;border-radius:6px;display:block}}
 figure.hero img{{outline:3px solid #c0392b}}
 figure.orphan img{{outline:2px dashed #e08e2b}}
 figcaption{{font-size:10px;color:#999;text-align:center;word-break:break-all}}
 .none{{color:#999;font-style:italic;font-size:13px}}
 .key{{background:#f6f9ff;border:1px solid #d6e4ff;border-radius:8px;padding:.8rem 1rem;font-size:14px}}
</style>
<h1>Scene clustering: time+geo fusion vs visual-only</h1>
<p class="key"><b>Noise bucket</b> = photos HDBSCAN could not confidently place into any scene on pixels
alone (label <code>&minus;1</code>). Fusing capture-time (the reliable backbone) and GPS (opportunistic)
into a single per-pair distance rescues many of these orphans into real scenes. "Rescued" counts photos
that were orphans in visual-only but landed in a real cluster once time+geo were added. No labels yet —
this shows the <i>direction and magnitude</i> of the change, not correctness.</p>
{summary}
<p class="hint"><b>Read the table:</b> noise (bold) falls on every trip; a few more clusters form. Adding
<i>time</i> does most of the work; <i>geo</i> refines. Germany (lowest GPS) improves most because time
carries it — the graceful-degradation claim, shown empirically.</p>
{"".join(galleries)}
<hr><p class="hint">Generated by <code>scripts/report_scene_fusion.py</code> (spec-103). Fuser:
<code>sim_bench/scene_cluster/geo_time_fusion.py</code>, weights w_v=w_t=w_g=1, time scale 30&nbsp;min,
geo scale 200&nbsp;m, HDBSCAN min_cluster_size=2 on the precomputed distance.</p>
"""


def build_summary_md(results: list[dict]) -> str:
    lines = ["# Scene fusion (spec-103) — before/after summary", "",
             "Visual-only vs time+geo fused scene clustering. Noise = unplaceable orphans (HDBSCAN -1).",
             "", "| Trip | Photos | GPS | Visual noise | +Time | +Time+Geo | Rescued |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f'| {r["trip"]} | {r["n"]} | {round(100*r["n_gps"]/r["n"])}% | '
            f'{r["visual"][1]} | {r["time"][1]} | {r["fused"][1]} | {len(r["rescued"])} |')
    lines += ["", "Time is the backbone (does most of the noise reduction); geo refines. Germany (lowest",
              "GPS) benefits most because time carries it — graceful degradation shown empirically.",
              "No reference labels yet: this shows change direction/magnitude, not correctness."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["budapest", "austria", "germany"])
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--slug", default="2026-07-21_scene_clustering_fusion")
    args = ap.parse_args()

    out_dir = REPORT_ROOT / args.slug
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    results = [run_trip(t, args.min_cluster_size, thumbs_dir) for t in args.trips]
    (out_dir / "report.html").write_text(build_html(results), encoding="utf-8")
    (out_dir / "summary.md").write_text(build_summary_md(results), encoding="utf-8")
    logger.warning("wrote %s", out_dir / "report.html")
    print(f"REPORT: {out_dir / 'report.html'}")
    for r in results:
        print(f"  {r['trip']}: noise {r['visual'][1]} -> {r['fused'][1]}, rescued {len(r['rescued'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
