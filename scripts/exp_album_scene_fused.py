"""Spec-103 — does build_scene_distance change the FINAL album picks? (the original question)

Runs the REAL Albumify arm on Budapest twice, identical except one inserts build_scene_distance before
cluster_scenes. Diffs: scene-cluster structure + the final K=20 selected images. Everything downstream
(select_best, coverage-first curation) is unchanged, so any difference in the picks is attributable to
the fused scene distance alone.

    .venv/Scripts/python scripts/exp_album_scene_fused.py --trip budapest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

WORK = Path(r"D:\albumify_vs_vlm")
OUT = Path(r"D:\sim-bench\reports\2026-07-21_scene_clustering_fusion")


def _n_scenes_noise(scene_clusters: dict) -> tuple[int, int]:
    real = {int(c): v for c, v in scene_clusters.items() if int(c) >= 0}
    noise = len(scene_clusters.get(-1, scene_clusters.get("-1", [])))
    return len(real), noise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="budapest")
    ap.add_argument("--pipeline", default="faces")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    from sim_bench.albumify_vs_vlm.albumify_arm import AlbumifyArmConfig, run_albumify_arm

    manifest = json.loads((WORK / args.trip / "manifest.json").read_text(encoding="utf-8"))
    imgs_dir = Path(manifest["out_dir"])
    ish = manifest["input_set_hash"]

    def run(fused: bool):
        cfg = AlbumifyArmConfig(target_k=args.k, pipeline=args.pipeline, fuse_scene_distance=fused)
        return run_albumify_arm(imgs_dir, args.trip, ish, cfg)

    print(f"=== {args.trip} / {args.pipeline} pipeline / K={args.k} ===", flush=True)
    print("running BASELINE (visual-only scene clustering) ...", flush=True)
    base = run(False)
    print("running FUSED (build_scene_distance inserted) ...", flush=True)
    fused = run(True)

    b_sc, b_noise = _n_scenes_noise(base.scene_clusters)
    f_sc, f_noise = _n_scenes_noise(fused.scene_clusters)
    base_picks, fused_picks = set(base.order), set(fused.order)
    common = base_picks & fused_picks
    added = sorted(fused_picks - base_picks)
    removed = sorted(base_picks - fused_picks)

    print(f"\nSCENE CLUSTERS:  baseline {b_sc} scenes / {b_noise} noise   ->   "
          f"fused {f_sc} scenes / {f_noise} noise")
    print(f"FINAL K={args.k} PICKS:  {len(common)}/{args.k} identical, "
          f"{len(added)} changed ({len(added)} in / {len(removed)} out)")
    print(f"  swapped IN  (fused chose, baseline didn't): {added}")
    print(f"  swapped OUT (baseline chose, fused didn't): {removed}")

    summary = {
        "trip": args.trip, "pipeline": args.pipeline, "k": args.k,
        "baseline": {"scenes": b_sc, "noise": b_noise, "order": base.order},
        "fused": {"scenes": f_sc, "noise": f_noise, "order": fused.order},
        "picks_identical": len(common), "changed": len(added),
        "swapped_in": added, "swapped_out": removed,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"album_fused_diff_{args.trip}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / f'album_fused_diff_{args.trip}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
