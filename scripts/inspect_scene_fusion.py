"""Spec-103 inspection — explain specific join/rescue decisions.

For a trip + a list of query stems, recompute the 3 variants (visual / +time / +time+geo) and print,
per query: its cluster id in each variant, and for the FUSED cluster it lands in, every member with the
time-gap (minutes) + geo-gap (metres) to the query and whether that member was an orphan in visual-only.
Answers "why did these join / why was this an orphan on pixels alone / is the time weight too strong".

    .venv/Scripts/python scripts/inspect_scene_fusion.py --trip budapest --stems 20250822_112331 ...
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
WORK = Path(r"D:\albumify_vs_vlm")


def _cluster(D, mcs=2):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="precomputed").fit_predict(D.astype(np.float64))


def main() -> int:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--time-scale", type=float, default=30.0)
    args = ap.parse_args()

    wd = WORK / args.trip
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
    gmeta = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo_meta = {recs[i]["stem"]: gmeta.get(srcs[i]) for i in range(len(recs))}

    from sim_bench.scene_cluster.geo_time_fusion import (SceneDistanceFuser, SceneDistanceInputs,
                                                         _time_from, _geo_from, _haversine_m)

    def lab(w_time, w_geo):
        f = SceneDistanceFuser(w_visual=1.0, w_time=w_time, w_geo=w_geo, time_scale_min=args.time_scale)
        D = f.calc(SceneDistanceInputs(emb, geo_meta, stems)).distance_matrix
        L = _cluster(D)
        return {stems[i]: int(L[i]) for i in range(len(stems))}, D

    Lv, Dv = lab(0.0, 0.0)
    Lt, _ = lab(1.0, 0.0)
    Lf, Df = lab(1.0, 1.0)
    idx = {s: i for i, s in enumerate(stems)}

    # cosine visual distance for reporting
    E = np.array([emb[s] / (np.linalg.norm(emb[s]) + 1e-12) for s in stems])
    cos = np.clip(E @ E.T, -1, 1)

    def times(s):
        return _time_from(s, geo_meta.get(s))

    for q in args.stems:
        if q not in idx:
            print(f"\n### {q}: NOT in {args.trip}")
            continue
        qt, qg = times(q), _geo_from(geo_meta.get(q))
        print(f"\n{'='*70}\n### {q}   time={qt}  geo={'yes' if qg else 'NONE'}")
        print(f"    cluster:  visual={Lv[q]:>3}   +time={Lt[q]:>3}   +time+geo={Lf[q]:>3}")
        fc = Lf[q]
        members = [s for s in stems if Lf[s] == fc and s != q] if fc != -1 else []
        print(f"    fused cluster {fc} has {len(members)+1} photos total. Members vs the query:")
        # sort members by time gap
        def tg(s):
            t = times(s)
            return abs((t - qt).total_seconds()) / 60.0 if (t and qt) else 9e9
        for s in sorted(members, key=tg):
            t = times(s)
            gap_min = tg(s)
            g = _geo_from(geo_meta.get(s))
            geo_m = _haversine_m(qg, g) if (qg and g) else None
            vis_d = 1 - cos[idx[q], idx[s]]
            fused_d = Df[idx[q], idx[s]]
            was_orphan = "ORPHAN-in-visual" if Lv[s] == -1 else f"was in visual-cluster {Lv[s]}"
            gm = f"{geo_m:6.0f}m" if geo_m is not None else "  no-geo"
            gp = f"{gap_min:6.1f}min" if gap_min < 8e9 else "  no-time"
            print(f"      {s}  dt={gp}  dgeo={gm}  visual_d={vis_d:.3f}  fused_d={fused_d:.3f}  [{was_orphan}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
