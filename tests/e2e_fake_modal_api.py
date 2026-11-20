"""The real API with a faked Modal, for Playwright (spec-116 W3, spec-117 W5): nothing reaches Modal or spends money.

The detached Modal call (``DetachedModalRemote``) and the photo Volume (``ModalVolumeRemote``) are replaced by one
stand-in: ``start`` runs the job's worker in a child process on this PC -- the same worker the L4 runs, which
writes its own progress, rows and files to the database -- and refuses to start when FAKE_MODAL_FAIL is set.
Everything else (the routes, JobService, ModalRunner following the job row, the rows) is the production code.

    python tests/e2e_fake_modal_api.py <port>
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


class FakeRemote:
    needs_shared_db = False           # the child worker reaches this PC's SQLite files

    def __init__(self, gpu: bool = True, **_):
        self.gpu, self._procs = gpu, {}

    def missing(self, keys):
        return list(keys)

    def upload(self, keys, store):
        time.sleep(2)                     # long enough for the page to show the stage

    def start(self, job_id, keys, spec, manifest, pipeline="default"):
        from sim_bench.serving.job_store import JobStore
        from sim_bench.serving.runners import LocalRunner
        from sim_bench.serving.serving_config import load_serving_config

        if os.environ.get("FAKE_MODAL_FAIL"):
            raise RuntimeError("Modal call failed: unreachable")
        store = JobStore(load_serving_config().jobs_root)
        runner = LocalRunner(store, pipeline=pipeline)
        self._procs[job_id] = subprocess.Popen(runner.command(job_id), env=runner.child_env())
        return job_id

    def status(self, call_id):
        proc = self._procs.get(call_id)
        return "running" if proc is not None and proc.poll() is None else "done"

    def cancel(self, call_id):
        proc = self._procs.get(call_id)
        if proc is not None:
            proc.kill()

    def delete_inputs(self, job_id, keys):
        return {"deleted": list(keys), "kept": 0}


def main() -> None:
    import uvicorn

    from sim_bench.serving import detached_modal, runners

    runners.ModalVolumeRemote = FakeRemote
    detached_modal.DetachedModalRemote = FakeRemote
    uvicorn.run("sim_bench.api.main:app", port=int(sys.argv[1]), log_level="warning")


if __name__ == "__main__":
    main()
