"""The GPU worker image has no FastAPI (spec-110/117): every module the worker imports, including the ones it imports
lazily inside a job, must load without it. 2026-11-14: `session.py` imported `fastapi.Request` at the top and every
L4 run failed at its first step with ModuleNotFoundError (found in spec-117 T8.2)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# what modal_app.py, curate_job.py and worker_persister.py import, at the top or inside a function
WORKER_MODULES = [
    "sim_bench.serving.modal_app", "sim_bench.serving.curate_job", "sim_bench.serving.worker_persister",
    "sim_bench.serving.db_progress", "sim_bench.serving.job_store", "sim_bench.serving.path_map",
    "sim_bench.serving.run_output", "sim_bench.serving.input_retention", "sim_bench.serving.pipeline_runner",
    "sim_bench.serving.run_settings", "sim_bench.serving.runners", "sim_bench.control.jobs",
    "sim_bench.api.database.models", "sim_bench.api.database.session", "sim_bench.api.services.run_persister",
    "sim_bench.pipeline.cache_handler", "sim_bench.pipeline.context", "sim_bench.pipeline.run",
    "sim_bench.album.normalizer", "sim_bench.album.results_bundle", "sim_bench.storage.database",
]

_PROBE = """
import importlib, sys
class _NoWebStack:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('fastapi', 'starlette', 'uvicorn'):
            raise ModuleNotFoundError(f"No module named {name!r} (not in the worker image)")
sys.meta_path.insert(0, _NoWebStack())
for m in sys.argv[1:]:
    importlib.import_module(m)
print('ok')
"""


def test_worker_modules_import_without_the_web_stack():
    done = subprocess.run([sys.executable, "-c", _PROBE, *WORKER_MODULES], cwd=REPO, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]
