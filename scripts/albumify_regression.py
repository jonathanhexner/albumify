"""Albumify regression runner (spec-113 R1): capture or check baselines on real albums.

    python scripts/albumify_regression.py capture quick budapest      # run + save baselines
    python scripts/albumify_regression.py check all                   # run + compare (exit 1 on a fail)
    python scripts/albumify_regression.py check quick --album <album.json>   # compare a finished album
    python scripts/albumify_regression.py check quick --job <job_id>        # compare a finished job

Each run goes through ``JobService`` (stage 768 px -> default pipeline -> load), the same path as
the product, into the app's own job list (control DB + job store), tagged ``regression:<dataset>``;
the baseline job is tagged ``regression:<dataset>:baseline``. Read the results in the Albumify app:
Compare page (spec-114). Runner is ``local`` (this machine, child process) unless ``--runner modal_gpu``.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

from sim_bench.album.regression import AlbumComparator, AlbumSummarizer, CompareInputs

logger = logging.getLogger("albumify_regression")

REPO = Path(__file__).resolve().parents[1]
REG_DIR = REPO / "tests" / "regression" / "albumify"
BASELINES = REG_DIR / "baselines"
TAG = "regression:{name}"
BASELINE_TAG = "regression:{name}:baseline"


def load_datasets() -> Dict[str, Dict]:
    return yaml.safe_load((REG_DIR / "datasets.yaml").read_text(encoding="utf-8"))


def _service():
    """The app's job service: same control DB ($DATABASE_URL or ~/.sim_bench/control.db) and job store."""
    from sim_bench.control.db import ControlDB
    from sim_bench.serving.job_service import JobService
    from sim_bench.serving.serving_config import load_serving_config

    return JobService(ControlDB(), load_serving_config())


def run_album(name: str, ds: Dict, runner: str) -> Dict:
    """Run one dataset through JobService; return its album.json plus run metadata."""
    from sim_bench.serving.job_service import SubmitInputs

    svc = _service()
    job_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    t0 = time.time()
    res = svc.submit(SubmitInputs(source=Path(ds["source"]), job_id=job_id, runner=runner,
                                  start=int(ds.get("start", 0)), n=ds.get("n"),
                                  tag=TAG.format(name=name)))
    if res.status != "succeeded":
        raise RuntimeError(f"{name}: job {job_id} {res.status}: {res.error}")
    album = svc.store.read_album(job_id)
    return {"album": album, "job_id": job_id, "wall_s": round(time.time() - t0, 1)}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def capture(name: str, ds: Dict, runner: str, album_path: Optional[Path]) -> None:
    run = ({"album": json.loads(album_path.read_text(encoding="utf-8")), "job_id": None,
            "wall_s": None} if album_path else run_album(name, ds, runner))
    baseline = {
        **AlbumSummarizer().calc(run["album"]),
        "meta": {"dataset": name, "source": ds["source"], "start": ds.get("start", 0),
                 "n": ds.get("n"), "runner": runner, "git": _git_sha(),
                 "captured": time.strftime("%Y-%m-%d %H:%M"), "job_id": run["job_id"],
                 "wall_s": run["wall_s"]},
    }
    BASELINES.mkdir(parents=True, exist_ok=True)
    out = BASELINES / f"{name}.json"
    out.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    if run["job_id"]:
        mark_baseline(name, run["job_id"])
    c = baseline["counts"]
    print(f"[capture] {name}: {c['kept']} picks / {c['uploaded']} in album, {c['scenes']} scenes, "
          f"{c['people']} people, {run['wall_s']}s -> {out}", flush=True)


def mark_baseline(name: str, job_id: str) -> None:
    """Tag ``job_id`` as the dataset's baseline; the previous baseline becomes an ordinary check run."""
    repo = _service().repo
    for job in repo.list(limit=10_000):
        if job.tag == BASELINE_TAG.format(name=name) and job.id != job_id:
            repo.set_tag(job.id, TAG.format(name=name))
    repo.set_tag(job_id, BASELINE_TAG.format(name=name))


def _existing_job(job_id: str) -> Dict:
    """A finished job, read back from the app's job store."""
    album = _service().store.read_album(job_id)
    if album is None:
        raise SystemExit(f"no album.json for job {job_id}")
    return {"album": album, "job_id": job_id, "wall_s": None}


def check(name: str, ds: Dict, runner: str, album_path: Optional[Path], job_id: Optional[str]) -> bool:
    path = BASELINES / f"{name}.json"
    if not path.exists():
        print(f"[check] {name}: no baseline at {path}; run `capture {name}` first")
        return False
    if job_id:
        run = _existing_job(job_id)
    elif album_path:
        run = {"album": json.loads(album_path.read_text(encoding="utf-8")), "job_id": None, "wall_s": None}
    else:
        run = run_album(name, ds, runner)
    baseline = json.loads(path.read_text(encoding="utf-8"))
    res = AlbumComparator().calc(CompareInputs(baseline=baseline,
                                               current=AlbumSummarizer().calc(run["album"])))
    status = "PASS" if res.ok else "FAIL"
    print(f"[check] {name}: {status}  {res.metrics}  ({run['wall_s']}s)", flush=True)
    for f in res.failures:
        print(f"    - {f}", flush=True)
    if run.get("job_id"):
        print(f"    see it: Albumify app -> Compare -> {name} (baseline {baseline['meta'].get('job_id')} "
              f"vs {run['job_id']})", flush=True)
    return res.ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "check"])
    ap.add_argument("datasets", nargs="+", help="names from datasets.yaml, or 'all'")
    ap.add_argument("--runner", default="local", choices=["local", "modal_gpu", "modal_cpu"])
    ap.add_argument("--album", type=Path, help="use this album.json instead of running (one dataset)")
    ap.add_argument("--job", help="check: use this finished regression job instead of running (one dataset)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    all_ds = load_datasets()
    names = list(all_ds) if args.datasets == ["all"] else args.datasets
    unknown = [n for n in names if n not in all_ds]
    if unknown:
        ap.error(f"unknown dataset(s): {unknown}; known: {list(all_ds)}")
    ok = True
    for n in names:
        if not Path(all_ds[n]["source"]).is_dir():
            print(f"[{args.mode}] {n}: source {all_ds[n]['source']} not found; skipped")
            ok = False
            continue
        if args.mode == "capture":
            capture(n, all_ds[n], args.runner, args.album)
        else:
            ok = check(n, all_ds[n], args.runner, args.album, args.job) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
