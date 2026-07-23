"""Spec-103 decision table — flat fusion (time-scale sweep) vs two-stage, per trip.

For each trip: compute DINOv2 embeddings + geo ONCE, then cluster many ways and print a table of
n_clusters / noise, plus the co-cluster verdict on the two hand-flagged reference pairs:

  BAD-MERGE  (budapest 20250822_193734 <-> 20250822_194159): visually different, 4.4 min apart.
             GOOD outcome = SPLIT (they should NOT be one scene).
  GOOD-RESCUE(austria  20230819_114651 <-> 20230819_114622): near-duplicate, 29 s apart, but flat
             visual-only orphaned one. GOOD outcome = JOIN.

The winning method splits the bad merge AND keeps the good rescue.

    .venv/Scripts/python scripts/exp_scene_sweep.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
WORK = Path(r"D:\albumify_vs_vlm")

# (trip, a, b, desired) — desired True = should co-cluster, False = should be separate
FLAGS = [
    ("budapest", "20250822_193734", "20250822_194159", False, "BAD-MERGE  (want SPLIT)"),
    ("austria", "20230819_114651", "20230819_114622", True, "GOOD-RESCUE (want JOIN)"),
]


def _hdbscan(D, mcs=2):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="precomputed").fit_predict(D.astype(np.float64))


def _stats(labels_map):
    clusters = {}
    for s, c in labels_map.items():
        clusters.setdefault(c, []).append(s)
    noise = len(clusters.pop(-1, []))
    return len(clusters), noise


def _load_trip(trip):
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
    return stems, emb, geo


def _co(labels_map, a, b):
    if a not in labels_map or b not in labels_map:
        return "n/a"
    la, lb = labels_map[a], labels_map[b]
    if la == -1 or lb == -1:
        return "split(noise)"
    return "JOINED" if la == lb else "split"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["budapest", "austria", "germany"])
    ap.add_argument("--scales", nargs="+", type=float, default=[5, 10, 15, 30])
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--gap", type=float, default=60.0)
    args = ap.parse_args()

    from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceFuser, SceneDistanceInputs
    from sim_bench.scene_cluster.two_stage import TwoStageInputs, TwoStageSceneClusterer

    summary = {}
    for trip in args.trips:
        stems, emb, geo = _load_trip(trip)
        rows = []

        # baseline: visual only (flat)
        Dv = SceneDistanceFuser(1.0, 0.0, 0.0).calc(SceneDistanceInputs(emb, geo, stems)).distance_matrix
        lm = {stems[i]: int(v) for i, v in enumerate(_hdbscan(Dv))}
        rows.append(("visual-only (flat)", *_stats(lm), lm))

        # flat fusion, time-scale sweep (w_time=w_geo=1)
        for sc in args.scales:
            D = SceneDistanceFuser(1.0, 1.0, 1.0, time_scale_min=sc).calc(
                SceneDistanceInputs(emb, geo, stems)).distance_matrix
            lm = {stems[i]: int(v) for i, v in enumerate(_hdbscan(D))}
            rows.append((f"flat fused, scale={sc:g}min", *_stats(lm), lm))

        # two-stage (visual sub-cluster within time/geo segments)
        ts = TwoStageSceneClusterer(gap_threshold_min=args.gap, visual_tau=args.tau).calc(
            TwoStageInputs(emb, geo, stems))
        rows.append((f"two-stage (gap={args.gap:g}m,tau={args.tau:g})",
                     *_stats(ts.labels), ts.labels))

        print(f"\n{'='*82}\n{trip.upper()}  ({len(stems)} photos)")
        print(f"{'method':38} {'clusters':>8} {'noise':>7}   flagged-pair verdicts")
        my_flags = [f for f in FLAGS if f[0] == trip]
        for name, nc, nn, lm in rows:
            verd = "  ".join(f"{tag.split()[0]}:{_co(lm, a, b)}" for (_, a, b, _, tag) in my_flags)
            print(f"{name:38} {nc:>8} {nn:>7}   {verd}")
        summary[trip] = [{"method": n, "clusters": nc, "noise": nn} for n, nc, nn, _ in rows]

    print("\nLEGEND: BAD-MERGE wants 'split'; GOOD-RESCUE wants 'JOINED'. Winner does both.")
    for _, a, b, want, tag in FLAGS:
        print(f"  {tag}: {a} <-> {b}")
    out = Path(r"D:\sim-bench\reports\2026-07-21_scene_clustering_fusion\sweep.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
