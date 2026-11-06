"""Detached L4 runs on the deployed Modal app (spec-117 R8, R9).

    start(job_id, keys, spec, manifest, pipeline) -> call id   spawn ``curate_job`` on the DEPLOYED app, by name
    status(call_id)  -> "running" | "done" | "failed: <why>" | "gone"
    cancel(call_id)

Nothing here holds a live link to the run: the call id is stored on the job row, the worker writes status,
progress and results to the database itself, so the PC can drop off the network or the API can restart and the
run goes on. ``modal`` is imported lazily: the API must start without Modal installed or logged in.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["DetachedModalRemote", "call_ended", "RUNNING", "DONE", "GONE"]

RUNNING, DONE, GONE = "running", "done", "gone"
APP_NAME = "albumify-curator"


class DetachedModalRemote:
    """Spawn / follow / cancel by call id. ``needs_shared_db``: the worker writes to the API's database, so that
    database must be one the L4 can reach (Neon), not a SQLite file on this PC."""

    needs_shared_db = True

    def __init__(self, gpu: bool = True, *, app_name: str = APP_NAME):
        self.gpu = gpu
        self.app_name = app_name if gpu else app_name + "-cpu"
        self.function = "curate_job" if gpu else "curate_job_cpu"

    def start(self, job_id: str, keys: Sequence[str], spec: Optional[Dict], manifest: Optional[Dict],
              pipeline: str = "default") -> str:
        import modal

        from sim_bench.serving.run_settings import current_commit

        fn = modal.Function.from_name(self.app_name, self.function)
        call = fn.spawn(job_id, pipeline, list(keys), current_commit(), spec, manifest)
        logger.info("job %s: spawned %s.%s, call %s", job_id, self.app_name, self.function, call.object_id)
        return call.object_id

    def status(self, call_id: str) -> str:
        import modal
        from modal.exception import OutputExpiredError

        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return DONE
        except TimeoutError:
            return RUNNING
        except OutputExpiredError:
            return GONE
        except Exception as e:          # cancelled, crashed, timed out on Modal's side
            return f"failed: {type(e).__name__}: {e}"

    def cancel(self, call_id: str) -> None:
        import modal

        modal.FunctionCall.from_id(call_id).cancel()


def call_ended(status: str) -> bool:
    """True once a call is no longer running (done, failed or gone)."""
    return status != RUNNING
