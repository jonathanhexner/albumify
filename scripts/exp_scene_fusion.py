"""Spec-103 quick look — does time/geo fusion change scene clustering vs visual-only?

Fast eyeball (NO labels, NO metrics): compute DINOv2 embeddings for a trip's working set, read GPS
+ time from the ORIGINAL photos (the 768px working set is EXIF-stripped), then cluster three ways and
print the before/after. Time from the YYYYMMDD_HHMMSS stem is the reliable backbone; GPS is added
where present. Uses the real spec-103 fuser so this shares code with the eventual pipeline step.

    .venv/Scripts/python scripts/exp_scene_fusion.py --trip budapest
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

WORK = Path(r"D:\albumify_vs_vlm")


def _cluster(D: np.ndarray, min_cluster_size: int = 2):
    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                             metric="precomputed").fit_predict(D.astype(np.float64))
    return labels


def _summarize(name: str, labels, ids):
    clusters = {}
    for i, lab in zip(ids, labels):
        clusters.setdefault(int(lab), []).append(i)
    noise = clusters.pop(-1, [])
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"\n[{name}]  clusters={len(clusters)}  noise={len(noise)}  sizes={sizes}")
    return {int(l): sorted(v) for l, v in clusters.items() if l != -1}, set(ids[k] for k in range(len(ids)) if labels[k] == -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="budapest")
    ap.add_argument("--min-cluster-size", type=int, default=2)
    args = ap.parse_args()

    wd = WORK / args.trip
    manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = manifest["records"]
    imgs_dir = Path(manifest["out_dir"])
    stems = [r["stem"] for r in recs]
    ws_paths = [str(imgs_dir / f"{s}.jpg") for s in stems]

    # --- DINOv2 embeddings (visual) ---
    print(f"computing DINOv2 embeddings for {len(stems)} images ...", flush=True)
    from sim_bench.feature_extraction.base import load_method
    extractor = load_method("dinov2", {"method": "dinov2", "batch_size": 32})
    feats = extractor.extract_features(ws_paths)
    emb = {stems[i]: feats[i] for i in range(len(stems))}

    # --- geo + time from ORIGINALS (working set is EXIF-stripped) ---
    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    srcs = [r["src"] for r in recs]
    gmeta_by_src = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo_meta = {recs[i]["stem"]: gmeta_by_src.get(srcs[i]) for i in range(len(recs))}
    n_gps = sum(1 for m in geo_meta.values() if m is not None and m.lat is not None)
    n_time = sum(1 for m in geo_meta.values() if m is not None and m.timestamp is not None)
    print(f"originals: {n_gps}/{len(stems)} have GPS, {n_time}/{len(stems)} have EXIF time "
          f"(+ filename fallback for the rest)")

    from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceFuser, SceneDistanceInputs

    def dist(w_time, w_geo):
        fuser = SceneDistanceFuser(w_visual=1.0, w_time=w_time, w_geo=w_geo)
        return fuser.calc(SceneDistanceInputs(emb, geo_meta, stems)).distance_matrix

    variants = {
        "visual only (baseline == today)": dist(0.0, 0.0),
        "+time":                           dist(1.0, 0.0),
        "+time +geo":                      dist(1.0, 1.0),
    }
    results = {}
    for name, D in variants.items():
        labels = _cluster(D, args.min_cluster_size)
        results[name] = _summarize(name, labels, stems)

    # --- how different is fused vs baseline? (pairs that co-cluster in one but not the other) ---
    base_clusters, _ = results["visual only (baseline == today)"]
    fused_clusters, _ = results["+time +geo"]
    base_pair = _copairs(base_clusters)
    fused_pair = _copairs(fused_clusters)
    only_base = base_pair - fused_pair
    only_fused = fused_pair - base_pair
    print(f"\n[diff visual vs +time+geo]  co-cluster pairs: base={len(base_pair)} fused={len(fused_pair)}")
    print(f"  pairs split apart by fusion : {len(only_base)}")
    print(f"  pairs newly joined by fusion: {len(only_fused)}")
    print("\n(eyeball the cluster size lists above; if fusion barely moves them, geo/time add little here)")
    return 0


def _copairs(clusters: dict) -> set:
    pairs = set()
    for members in clusters.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
