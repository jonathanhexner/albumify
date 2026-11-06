"""spec-117 R10: Modal tries L4 then A10G; the spend cap is checked at the dearest GPU; the run records its GPU."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PIL import Image

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.serving.job_service import JobService
from sim_bench.serving.serving_config import ServingConfig, load_serving_config

MODAL_APP = Path(__file__).resolve().parents[2] / "sim_bench" / "serving" / "modal_app.py"


def test_the_gpu_list_is_l4_then_a10g():
    cfg = load_serving_config()
    assert cfg.gpu == ("L4", "A10G")
    assert cfg.gpu_dollars_per_hour == {"L4": 0.80, "A10G": 1.10}
    assert cfg.dearest_dollars_per_hour == 1.10


def test_the_spend_cap_is_checked_at_the_dearest_gpu(tmp_path, control_db_url):
    src = tmp_path / "trip"
    src.mkdir()
    for i in range(100):
        Image.new("RGB", (8, 8)).save(src / f"{i:03d}.jpg")
    cfg = load_serving_config()
    est = JobService(ControlDB(control_db_url), ServingConfig(jobs_root=tmp_path / "jobs")).estimate(
        src, runner="modal_gpu")
    minutes = round((cfg.l4_fixed_seconds + 100 * cfg.l4_seconds_per_photo) / 60, 1)
    assert est.gpu_minutes == minutes
    assert est.dollars == round(minutes / 60 * 1.10, 2)          # A10G, not L4


def test_an_unknown_gpu_is_priced_at_the_l4_rate_and_an_empty_list_is_refused():
    assert ServingConfig(gpu=("H100",), gpu_dollars_per_hour={}).dearest_dollars_per_hour == 0.80
    with pytest.raises(ValueError):
        ServingConfig(gpu=())


def _decorator_kwargs(fn_name: str) -> dict:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    call = fn.decorator_list[0]
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_the_job_functions_ask_for_the_configured_list_and_the_secret():
    """The deployed functions: gpu=GPU_TYPES (from serving.yaml), secret albumify-cloud, no cache Volume."""
    gpu_job = _decorator_kwargs("curate_job")
    assert gpu_job["gpu"] == "GPU_TYPES"
    assert "CLOUD_SECRET" in gpu_job["secrets"]
    assert "CLOUD_SECRET" in _decorator_kwargs("curate_job_cpu")["secrets"]
    src = MODAL_APP.read_text(encoding="utf-8")
    assert "GPU_TYPES = list(load_serving_config().gpu)" in src
    assert 'CLOUD_SECRET = "albumify-cloud"' in src
    assert "albumify-cache" not in src and "FEATURE_CACHE_ENV" not in src


def test_the_run_records_the_gpu_it_got(tmp_path, control_db_url):
    """The worker writes the GPU name on the job row and in run.json."""
    from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
    from sim_bench.serving.job_store import JobStore
    from tests.serving.test_run_output import RESULT, finished_context

    db = ControlDB(control_db_url)
    repo = JobRepo(db)
    repo.create("j1", runner="modal_gpu")
    repo.set_progress("j1", None, gpu="NVIDIA A10G")
    assert repo.get("j1").gpu == "NVIDIA A10G"

    store = JobStore(tmp_path / "jobs")
    store.inputs.put_bytes("00000000000000aa/a.jpg", b"x")
    store.write_input_manifest("j1", ["00000000000000aa/a.jpg"])
    JobCurator(store, runner=lambda *a, **k: (finished_context(), RESULT)).calc(
        CurateJobInputs(job_id="j1", runner="modal_gpu", gpu="NVIDIA A10G"))
    import json
    assert json.loads(store.paths("j1").run_path.read_text(encoding="utf-8"))["gpu"] == "NVIDIA A10G"
