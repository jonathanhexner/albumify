"""spec-116 T4.1 diagnostic: full-size vs 768 with the sharpness gate OFF on both arms (NOT the production default).

The default-config comparison (spec116_r13_fullsize_vs_768.py) is dominated by ``filter_quality.min_sharpness``: the
score depends on resolution, so one threshold removes 114 of 122 full-size photos and 26 of 122 copies. This re-runs
both arms from that experiment's folder and warm feature cache with the threshold at 0, to see the rest of the
pipeline on its own. It re-uses the full-size job id (the cache is keyed by path), keeping that job's default-config
results in ``results_default/``.

Run after the main script: .venv/Scripts/python scripts/spec116_r13_sharpness_diagnostic.py --out <report folder>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / ".sim_bench" / "spec116_r13"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    os.environ.update(DATABASE_URL=f"sqlite:///{(data / 'control.db').as_posix()}", ALBUMIFY_JOBS_ROOT=str(data / "jobs"),
                      ALBUMIFY_FEATURE_CACHE=str(data / "cache_122.db"), SIM_BENCH_DB=str(data / "sim_bench.db"))

    from sim_bench.control.compare import CompareJobsInputs, JobComparer
    from sim_bench.control.db import ControlDB
    from sim_bench.serving.job_service import JobService, SubmitInputs
    from sim_bench.serving.serving_config import load_serving_config

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    jobs = JobService(ControlDB(), load_serving_config())
    spec = copy.deepcopy(jobs.store.read_run_spec(results["run_768"]))
    spec["step_configs"]["filter_quality"]["min_sharpness"] = 0.0

    full, copies = "budapest-fullsize", "budapest-768-nogate"
    keep = jobs.store.paths(full).root / "results_default"
    if not keep.exists():
        shutil.copytree(jobs.store.paths(full).results_dir, keep)

    inputs = SubmitInputs(job_id=full, runner="local", pipeline="default", run_spec=spec)
    jobs.create(inputs)
    res = jobs.run(full, inputs)
    if res.error:
        raise SystemExit(f"full-size arm failed: {res.error}")
    inputs = SubmitInputs(job_id=copies, runner="local", pipeline="default", run_spec=spec, rerun_of=results["run_768"])
    jobs.create(inputs)
    res = jobs.run(copies, inputs)
    if res.error:
        raise SystemExit(f"768 arm failed: {res.error}")

    cmp = JobComparer(jobs.db).calc(CompareJobsInputs(a=full, b=copies)).to_dict()
    (out / "diagnostic_no_sharpness_gate.json").write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")
    print("DONE ok", cmp["ok"], cmp["failures"], cmp["metrics"], {k: len(v) for k, v in cmp["picks"].items()})


if __name__ == "__main__":
    main()
