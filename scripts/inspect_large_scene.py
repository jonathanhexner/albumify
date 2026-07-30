"""Why is a scene big? Print its time span + visual 'diameter' to detect single-linkage chaining."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

WORK = Path(r"D:\albumify_vs_vlm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--rank", type=int, default=1, help="1 = largest scene, 2 = 2nd largest ...")
    args = ap.parse_args()

    wd = WORK / args.trip
    manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = manifest["records"]; imgs = Path(manifest["out_dir"])
    stems = [r["stem"] for r in recs]
    from sim_bench.feature_extraction.base import load_method
    feats = load_method("dinov2", {"method": "dinov2", "batch_size": 32}).extract_features(
        [str(imgs / f"{s}.jpg") for s in stems])
    emb = {stems[i]: feats[i] for i in range(len(stems))}
    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    srcs = [r["src"] for r in recs]
    gm = GeoMetadataExtractor().calc(ExifInputs(image_paths=srcs)).metadata
    geo = {recs[i]["stem"]: gm.get(srcs[i]) for i in range(len(recs))}
    from sim_bench.scene_cluster.two_stage import TwoStageInputs, TwoStageSceneClusterer
    from sim_bench.scene_cluster.geo_time_fusion import _time_from
    ts = TwoStageSceneClusterer(gap_threshold_min=60, visual_tau=0.40, burst_sec=60, burst_tau=0.65).calc(
        TwoStageInputs(emb, geo, stems))

    clusters = {}
    for s, c in ts.labels.items():
        if c != -1:
            clusters.setdefault(c, []).append(s)
    ranked = sorted(clusters.values(), key=len, reverse=True)
    scene = sorted(ranked[args.rank - 1])
    times = [_time_from(s, geo.get(s)) for s in scene]
    tvals = [t for t in times if t]
    E = np.array([emb[s] / (np.linalg.norm(emb[s]) + 1e-12) for s in scene])
    D = 1 - np.clip(E @ E.T, -1, 1)
    iu = np.triu_indices(len(scene), 1)
    pair = D[iu]
    print(f"\n{args.trip} scene rank {args.rank}: {len(scene)} photos")
    if tvals:
        span = (max(tvals) - min(tvals)).total_seconds() / 60.0
        print(f"time span: {min(tvals)}  ->  {max(tvals)}   ({span:.0f} min)")
    print(f"visual distance within scene:  min={pair.min():.2f}  median={np.median(pair):.2f}  "
          f"MAX(diameter)={pair.max():.2f}")
    print(f"  (diameter >> tau=0.40 means single-linkage CHAINING: A~B~C daisy-chained though A and C differ)")
    # farthest pair
    a, b = iu[0][pair.argmax()], iu[1][pair.argmax()]
    print(f"  farthest-apart pair in this one scene: {scene[a]}  <->  {scene[b]}  (visual_d={pair.max():.2f})")
    print(f"first/last few: {scene[:3]} ... {scene[-3:]}")


if __name__ == "__main__":
    raise SystemExit(main())
