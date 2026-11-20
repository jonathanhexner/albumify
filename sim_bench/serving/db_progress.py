"""A run's live state in its job row (spec-117 R5), written by the worker; replaces ``progress.json``.

    jobs.progress     {"stage": "running", "current_step": "triage_content", "step_fraction": 0.43,
                       "photos_done": 307, "photos_total": 715, "completed_steps": [{step, duration_ms, ...}]}
    jobs.heartbeat_at  refreshed every ``heartbeat_s`` by a background thread, and with every write

The app reads only this row, whichever machine runs the job, so a run on the L4 shows the same in-step
percentage as one on this PC. In-step fractions come from the steps' ``context.report_progress`` and are
written at most every ``min_interval_s``; step ends and the heartbeat always are. A failed write is logged and
dropped: progress must never stop the run it describes.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Dict, List, Optional

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo

logger = logging.getLogger(__name__)

__all__ = ["DbProgressWriter", "NullProgress"]

_N_OF_M = re.compile(r"(\d+)\s*/\s*(\d+)")


class DbProgressWriter:
    """Config in ``__init__``; the pipeline calls ``on_step`` / ``on_fraction``; ``start`` / ``stop`` the heartbeat."""

    def __init__(self, db: ControlDB, job_id: str, steps: List[str], *, total: int = 0,
                 min_interval_s: float = 2.0, heartbeat_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self.job_id, self.db = job_id, db
        self._repo = JobRepo(db)
        self._steps, self._total = list(steps), int(total)
        self._min_interval, self._heartbeat_s, self._clock = min_interval_s, heartbeat_s, clock
        self._done: List[Dict] = []
        self._fraction: Optional[float] = None
        self._n_done: Optional[int] = None
        self._current_override: Optional[str] = None
        self._last_write = float("-inf")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- what the app sees --------------------------------------------------

    def snapshot(self) -> Dict:
        names = {d["step"] for d in self._done}
        current = self._current_override or next((s for s in self._steps if s not in names), None)
        out: Dict = {"stage": "running", "current_step": current, "completed_steps": list(self._done)}
        if self._fraction is not None:
            out["step_fraction"] = round(self._fraction, 4)
        if self._total:
            out["photos_total"] = self._total
            if self._n_done is not None:
                out["photos_done"] = self._n_done
        return out

    # --- hooks ----------------------------------------------------------------

    def on_step(self, step_result) -> None:
        """``execute_spec(on_step_complete=...)``: one finished step, always written."""
        with self._lock:
            self._done.append({"step": step_result.step_name, "duration_ms": step_result.duration_ms,
                               "status": "completed" if step_result.success else "failed",
                               "error": step_result.error_message})
            self._fraction = self._n_done = self._current_override = None
            self._write_locked()

    def on_fraction(self, step: str, fraction: float, message: str = "", done: Optional[int] = None,
                    total: Optional[int] = None) -> None:
        """``context.report_progress`` (step, 0..1, message): written at most every ``min_interval_s``."""
        with self._lock:
            self._current_override = step
            self._fraction = max(0.0, min(1.0, float(fraction)))
            m = _N_OF_M.search(message or "")
            if done is None and m:
                done, total = int(m.group(1)), int(m.group(2))
            if total:
                self._total = int(total)
            self._n_done = int(done) if done is not None else (
                round(self._fraction * self._total) if self._total else None)
            if self._fraction < 1.0 and self._clock() - self._last_write < self._min_interval:
                return
            self._write_locked()

    def heartbeat(self) -> None:
        try:
            self._repo.set_progress(self.job_id, None, heartbeat=True)
        except Exception as e:      # never stops the run
            logger.warning("job %s: heartbeat not written: %s", self.job_id, type(e).__name__)

    def set_stage(self, stage: str) -> None:
        """A stage outside the steps (e.g. "saving results"), always written."""
        with self._lock:
            snap = {**self.snapshot(), "stage": stage}
            self._write(snap)

    def _write_locked(self) -> None:
        self._write(self.snapshot())

    def _write(self, progress: Dict) -> None:
        self._last_write = self._clock()
        try:
            self._repo.set_progress(self.job_id, progress, heartbeat=True)
        except Exception as e:
            logger.warning("job %s: progress not written: %s", self.job_id, type(e).__name__)

    # --- heartbeat thread -------------------------------------------------------

    def start(self) -> "DbProgressWriter":
        with self._lock:
            self._write_locked()
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name=f"heartbeat-{self.job_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._heartbeat_s + 1)
            self._thread = None

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            self.heartbeat()

    def __enter__(self) -> "DbProgressWriter":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class NullProgress:
    """No database (unit tests, a worker without one): the same hooks, nothing written."""

    def on_step(self, step_result) -> None:
        pass

    def on_fraction(self, step: str, fraction: float, message: str = "", **_) -> None:
        pass

    def heartbeat(self) -> None:
        pass

    def set_stage(self, stage: str) -> None:
        pass

    def start(self) -> "NullProgress":
        return self

    def stop(self) -> None:
        pass

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *exc) -> None:
        pass
