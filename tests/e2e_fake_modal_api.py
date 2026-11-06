"""The real API with a faked Modal, for Playwright (spec-116 W3): nothing here reaches Modal or spends money.

``ModalVolumeRemote`` is replaced by a stand-in that runs the job in a child process on this PC, then rewrites the
result as if the worker had read ``/inputs/...``, and refuses to start when FAKE_MODAL_FAIL is set. Everything else
(the routes, JobService, ModalRunner, the remap, the rows) is the production code.

    python tests/e2e_fake_modal_api.py <port>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


class FakeRemote:
    def __init__(self, gpu: bool = True):
        self.gpu, self._proc, self._cancelled = gpu, None, False

    def missing(self, keys):
        return list(keys)

    def upload(self, keys, store):
        time.sleep(2)                     # long enough for the page to show the stage

    def run_job(self, job_id, keys, pipeline, spec=None):
        from sim_bench.serving.job_store import JobStore
        from sim_bench.serving.runners import CANCELLED, LocalRunner
        from sim_bench.serving.serving_config import load_serving_config

        if os.environ.get("FAKE_MODAL_FAIL"):
            return {"ok": False, "error": "Modal call failed: unreachable"}
        store = JobStore(load_serving_config().jobs_root)
        self._proc = subprocess.Popen(LocalRunner(store, pipeline=pipeline).command(job_id))
        self._proc.wait()
        if self._cancelled:
            return {"ok": False, "error": CANCELLED}
        path = store.paths(job_id).run_output_path
        path.write_text(json.dumps(_as_worker_paths(json.loads(path.read_text(encoding="utf-8")),
                                                    str(store.inputs.root))), encoding="utf-8")
        return {"ok": True}

    def cancel(self):
        self._cancelled = True
        if self._proc is not None:
            self._proc.kill()

    def download_results(self, job_id, store):
        return 3


def _as_worker_paths(value, root: str):
    if isinstance(value, str):
        return value.replace(root + os.sep, "/inputs/").replace("\\", "/") if value.startswith(root) else value
    if isinstance(value, list):
        return [_as_worker_paths(v, root) for v in value]
    if isinstance(value, dict):
        return {_as_worker_paths(k, root): _as_worker_paths(v, root) for k, v in value.items()}
    return value


def main() -> None:
    import uvicorn

    from sim_bench.serving import runners

    runners.ModalVolumeRemote = FakeRemote
    uvicorn.run("sim_bench.api.main:app", port=int(sys.argv[1]), log_level="warning")


if __name__ == "__main__":
    main()
