"""spec-116 T4.1 (R13, R9 on this PC): does analysing the 768 px copies pick close enough to the full-size originals?

Everything lands in its own folder (--data, default ~/.sim_bench/spec116_r13): app DB, control DB, job store and
feature caches. The real ~/.sim_bench databases are not touched.

    768 arm        the production path: PipelineService -> JobService -> child process -> RunPersister
    full-size arm  the same RunSpec, on a job whose input/ holds the original files (the old path's inputs; the old
                   in-API runner is gone, and this keeps the two arms in the same code)
    PC re-run      48 of the photos, cold cache then warm cache (R9)

Run: .venv/Scripts/python scripts/spec116_r13_fullsize_vs_768.py --out reports/2026-10-30_spec116_r13_fullsize_vs_768
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path


def _arm(svc, album_id: str, steps, config) -> tuple:
    run_id = svc.start_pipeline(album_id, steps=steps, step_configs=dict(config))
    start = time.time()
    svc.execute_pipeline(run_id)
    wall = round(time.time() - start, 1)
    run = svc.get_status(run_id)
    if run.status != "completed":
        raise SystemExit(f"run {run_id} failed: {run.error_message}")
    return run_id, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="D:/Budapest2025_Google")
    ap.add_argument("--data", default=str(Path.home() / ".sim_bench" / "spec116_r13"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    if data.exists():
        raise SystemExit(f"{data} exists: this experiment starts clean, move it away first")
    data.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(SIM_BENCH_DB=str(data / "sim_bench.db"), DATABASE_URL=f"sqlite:///{(data / 'control.db').as_posix()}",
                      ALBUMIFY_JOBS_ROOT=str(data / "jobs"), ALBUMIFY_FEATURE_CACHE=str(data / "cache_122.db"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # imported after the environment points at the experiment's folder
    from sim_bench.api.database.session import get_session_direct, init_db
    from sim_bench.api.services.album_service import AlbumService
    from sim_bench.api.services.config_service import ConfigService
    from sim_bench.api.services.pipeline_service import PipelineService
    from sim_bench.control.compare import CompareJobsInputs, JobComparer
    from sim_bench.control.db import ControlDB
    from sim_bench.serving.job_service import JobService, SubmitInputs
    from sim_bench.serving.serving_config import load_serving_config
    from sim_bench.serving.staging import select_images

    init_db()
    session = get_session_direct()
    jobs = JobService(ControlDB(), load_serving_config())
    svc = PipelineService(session, job_service=jobs)
    config = ConfigService(session).get_merged_config()
    steps = config["default_pipeline"]
    album = AlbumService(session).create("Budapest122", args.source)
    results = {"source": args.source, "n_steps": len(steps)}

    # --- 768 arm: the production path
    run_768, results["wall_768_s"] = _arm(svc, album.id, steps, config)
    results["run_768"] = run_768

    # --- full-size arm: same spec, the original files as inputs
    photos = select_images(Path(args.source))
    results["n_photos"] = len(photos)
    full = "budapest-fullsize"
    jobs.store.stage_inputs(full, [(p.name, p.read_bytes()) for p in photos])
    inputs = SubmitInputs(job_id=full, runner="local", pipeline="default", run_spec=jobs.store.read_run_spec(run_768))
    jobs.create(inputs)
    start = time.time()
    res = jobs.run(full, inputs)
    results["wall_full_s"] = round(time.time() - start, 1)
    if res.error:
        raise SystemExit(f"full-size arm failed: {res.error}")

    results["compare"] = JobComparer(jobs.db).calc(CompareJobsInputs(a=full, b=run_768)).to_dict()
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # --- R9 on this PC: 48 photos, cold then warm feature cache
    folder = data / "photos48"
    folder.mkdir()
    for p in photos[:48]:
        shutil.copy2(p, folder / p.name)
    os.environ["ALBUMIFY_FEATURE_CACHE"] = str(data / "cache_48.db")
    album48 = AlbumService(session).create("Budapest48", str(folder))
    results["r9_first_run"], results["r9_first_wall_s"] = _arm(svc, album48.id, steps, config)
    results["r9_rerun"], results["r9_rerun_wall_s"] = _arm(svc, album48.id, steps, config)
    results["r9_ratio"] = round(results["r9_rerun_wall_s"] / results["r9_first_wall_s"], 3)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("DONE", {k: v for k, v in results.items() if k != "compare"})


if __name__ == "__main__":
    main()
