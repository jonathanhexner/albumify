"""Drive the Modal curation worker on a small local image set (spec-110 AC3/AC4/AC5 proof).

Two modes:

  inline (the original scaffold) — sends image bytes in the call, prints a pipeline summary:
      .venv/Scripts/python scripts/modal_curate_driver.py --src D:/Slovakia26 --start 300 --n 12 --cpu
      .venv/Scripts/python scripts/modal_curate_driver.py --src D:/Slovakia26 --start 300 --n 12   # GPU

  job (spec-110 T2.1, storage-reference) — stages the images into the job store, curates by
  job_id, then reads album.json back out. `--twice` re-runs curate on the same job and diffs the
  two albums (AC4 idempotence):
      .venv/Scripts/python scripts/modal_curate_driver.py --src D:/Slovakia26 --n 12 --job-id slovakia12 --cpu --twice

  local (no Modal, no spend) -- the same JobStore/JobCurator path under a local directory, which is
  how the storage-reference contract and AC4 are proven on the dev box:
      .venv/Scripts/python scripts/modal_curate_driver.py --src D:/Slovakia26 --start 300 --n 12           --job-id localproof12 --pipeline minimal --local-root reports/_local_jobs --twice

Windows: set PYTHONUTF8=1 first. Images are downsampled to 768 px longest edge (the pipeline's
analysis tier) with EXIF preserved (triage_metadata needs it).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim_bench.serving.staging import downsample  # noqa: E402  (one staging path, spec-112)


def _log_timing(timing: dict, n_images: int) -> None:
    total = (timing or {}).get("total_ms") or 0
    if not total:
        return
    logger.info("pipeline total %.1fs = %.2f s/img", total / 1000, total / 1000 / max(1, n_images))
    for st in sorted(timing.get("steps") or [], key=lambda s: -(s.get("ms") or 0))[:8]:
        logger.info("  %-28s %7.0f ms  (%.2f s/img)", st["step"], st["ms"],
                    (st["ms"] or 0) / 1000 / max(1, n_images))


def run_inline(args, images) -> dict:
    """Original path: bytes in the call, pipeline summary out. No album.json is written."""
    import modal
    from sim_bench.serving.modal_app import app, cpu_app, curate, curate_cpu

    fn, run_app = (curate_cpu, cpu_app) if args.cpu else (curate, app)
    t0 = time.time()
    with modal.enable_output(), run_app.run():
        res = fn.remote(images, args.pipeline)
    res["driver"] = {"mode": "inline-cpu" if args.cpu else "inline-gpu", "n_images": len(images),
                     "wall_s": round(time.time() - t0, 1), "inputs": [n for n, _ in images]}

    logger.info("ok=%s wall=%.0fs", res.get("ok"), res["driver"]["wall_s"])
    if res.get("ok"):
        logger.info("selected (%d): %s", len(res["selected"]), [Path(p).name for p in res["selected"]])
        logger.info("set_aside: %s", [(Path(e["image"]).name, e.get("reason")) if isinstance(e, dict) else e
                                      for e in res["set_aside"]])
        _log_timing(res.get("timing"), len(images))
    else:
        logger.error("failed_step=%s error=%s", res.get("failed_step"), res.get("error"))
    return res


def run_job(args, images) -> dict:
    """spec-110 T2.1: stage -> curate(job_id) -> fetch album.json. Optionally curate twice (AC4)."""
    import modal
    from sim_bench.serving.curate_job import stable_album
    from sim_bench.serving.modal_app import (app, cpu_app, curate_job, curate_job_cpu, fetch_album,
                                             stage_job)

    job_id = args.job_id
    out: dict = {"job_id": job_id, "driver": {"mode": "job-cpu" if args.cpu else "job-gpu",
                                              "n_images": len(images),
                                              "inputs": [n for n, _ in images]}}

    t0 = time.time()
    with modal.enable_output(), cpu_app.run():
        out["staged"] = stage_job.remote(job_id, images)
    logger.info("staged %d images into %s", len(out["staged"]["staged"]), out["staged"]["input_dir"])

    fn, run_app = (curate_job_cpu, cpu_app) if args.cpu else (curate_job, app)
    runs = []
    with modal.enable_output(), run_app.run():
        for i in range(2 if args.twice else 1):
            t1 = time.time()
            res = fn.remote(job_id, args.pipeline)
            res["wall_s"] = round(time.time() - t1, 1)
            logger.info("run %d: ok=%s wall=%.0fs", i + 1, res.get("ok"), res["wall_s"])
            runs.append(res)
    out["runs"] = runs

    with modal.enable_output(), cpu_app.run():
        out["fetched"] = fetch_album.remote(job_id)
    out["driver"]["wall_s"] = round(time.time() - t0, 1)

    first = runs[0]
    if not first.get("ok"):
        logger.error("failed_step=%s error=%s", first.get("failed_step"), first.get("error"))
        return out

    album = first["album"]
    logger.info("album.json -> %s", first["album_path"])
    logger.info("counts: %s", album["counts"])
    logger.info("picks (%d): %s", len(album["picks"]), [p["id"] for p in album["picks"]])
    for d in album["dropped"]:
        logger.info("  dropped %-28s %-14s %s", d["id"], d["reason"], d["reason_detail"])
    logger.info("displays written: %s", out["fetched"]["displays"])
    logger.info("device flipped on %d steps: %s", len(first["flipped"]), first["flipped"])
    _log_timing(first.get("timing"), len(images))

    # the store must hold exactly what the worker returned
    out["album_matches_store"] = stable_album(out["fetched"]["album"]) == stable_album(album)
    logger.info("album.json in store matches the returned album: %s", out["album_matches_store"])

    if args.twice:
        out["idempotent"] = stable_album(runs[0]["album"]) == stable_album(runs[1]["album"])
        logger.info("AC4 idempotence (same job_id re-run, created_at excluded): %s", out["idempotent"])
        if not out["idempotent"]:
            a, b = runs[0]["album"], runs[1]["album"]
            logger.error("picks run1=%s run2=%s", [p["id"] for p in a["picks"]],
                         [p["id"] for p in b["picks"]])
    return out


def run_local(args, images) -> dict:
    """Same job path with no Modal at all: stage + curate under a local root.

    This is how the storage-reference contract is proven (and re-proven) without spend — the only
    difference from the worker is where the job directory lives.
    """
    from sim_bench.serving.curate_job import CurateJobInputs, JobCurator, stable_album
    from sim_bench.serving.job_store import JobStore

    store = JobStore(args.local_root)
    job_id = args.job_id
    staged = store.stage_inputs(job_id, images)
    logger.info("staged %d images into %s", len(staged), store.paths(job_id).input_dir)

    curator = JobCurator(store)
    runs = []
    for i in range(2 if args.twice else 1):
        t0 = time.time()
        res = curator.calc(CurateJobInputs(job_id=job_id, pipeline=args.pipeline, device="cpu"))
        runs.append({**{k: getattr(res, k) for k in ("ok", "n_inputs", "album", "album_path",
                                                     "flipped", "timing", "failed_step", "error")},
                     "wall_s": round(time.time() - t0, 1)})
        logger.info("run %d: ok=%s wall=%.0fs", i + 1, res.ok, runs[-1]["wall_s"])

    out = {"job_id": job_id, "runs": runs,
           "driver": {"mode": "local", "n_images": len(images), "root": str(args.local_root),
                      "inputs": [n for n, _ in images]}}
    first = runs[0]
    if not first["ok"]:
        logger.error("failed_step=%s error=%s", first["failed_step"], first["error"])
        return out

    album = first["album"]
    logger.info("album.json -> %s", first["album_path"])
    logger.info("counts: %s", album["counts"])
    logger.info("picks (%d): %s", len(album["picks"]), [p["id"] for p in album["picks"]])
    for d in album["dropped"]:
        logger.info("  dropped %-28s %-14s %s", d["id"], d["reason"], d["reason_detail"])
    _log_timing(first.get("timing"), len(images))
    if args.twice:
        out["idempotent"] = stable_album(runs[0]["album"]) == stable_album(runs[1]["album"])
        logger.info("AC4 idempotence (created_at excluded): %s", out["idempotent"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pipeline", default="default")
    ap.add_argument("--cpu", action="store_true", help="use the CPU app (free tier) instead of GPU")
    ap.add_argument("--job-id", default=None,
                    help="storage-reference mode: stage into the job store and curate by id")
    ap.add_argument("--twice", action="store_true",
                    help="job mode: curate the same job twice and diff the albums (AC4)")
    ap.add_argument("--local-root", type=Path, default=None,
                    help="run the job path locally under this dir (no Modal, no spend)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.twice and not args.job_id:
        ap.error("--twice requires --job-id (idempotence is a property of the stored job)")
    if args.local_root and not args.job_id:
        ap.error("--local-root requires --job-id (the local run uses the same job store layout)")

    files = sorted(args.src.glob("*.jpg"))[args.start:args.start + args.n]
    if not files:
        logger.error("no .jpg files under %s (start=%d n=%d)", args.src, args.start, args.n)
        return 2
    images = [(p.name, downsample(p)) for p in files]
    logger.info("sending %d images, %.1f MB", len(images), sum(len(b) for _, b in images) / 1e6)

    if args.local_root:
        res = run_local(args, images)
    elif args.job_id:
        res = run_job(args, images)
    else:
        res = run_inline(args, images)

    prefix = "_local_job_" if args.local_root else "_modal_job_"
    default_name = f"{prefix}{args.job_id}.json" if args.job_id else "_modal_curate_last.json"
    out = args.out or (REPO / "reports" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    logger.info("full result: %s", out)

    runs = res.get("runs") or [res]
    return 0 if all(r.get("ok") for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
