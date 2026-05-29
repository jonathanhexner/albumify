"""spec-045 §"Locked decisions" #4 — typed async handle for heavy compute.

Lifts the legacy ``app/face_clustering/state.py::_AsyncState`` polling pattern
into a typed, generic, framework-level primitive. The tab layer polls
:meth:`AsyncHandle.poll` and inspects :attr:`AsyncHandle.state` /
:attr:`AsyncHandle.result` / :attr:`AsyncHandle.error` — no raw
``threading.Thread`` ever leaks into the UI.

The legacy semantics are preserved verbatim:

* one daemon thread per ``start()`` call;
* ``poll()`` is non-blocking and idempotent;
* ``cancel()`` flips the state but does not kill the in-flight thread
  (Python threads can't be interrupted from outside) — the consumer should
  treat the result as discarded.

Future tabs that need heavy compute (Face Analysis, Recluster) reuse this
class — it is shared library code, not per-tab.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Generic, Literal, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

State = Literal["pending", "running", "done", "failed", "cancelled"]


@dataclass
class AsyncHandle(Generic[T]):
    """One in-flight (or finished) background computation.

    Construct via :meth:`AsyncHandle.start`. Inspect :attr:`state` after
    each :meth:`poll`; read :attr:`result` only when ``state == "done"``,
    :attr:`error` only when ``state == "failed"``.
    """

    state: State = "pending"
    result: Optional[T] = None
    error: Optional[BaseException] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def start(cls, fn: Callable[..., T], *args, **kwargs) -> "AsyncHandle[T]":
        """Start ``fn(*args, **kwargs)`` in a daemon thread and return the handle."""
        handle: "AsyncHandle[T]" = cls(state="running")

        def _runner() -> None:
            try:
                value = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                with handle._lock:
                    if handle.state != "cancelled":
                        handle.error = exc
                        handle.state = "failed"
                logger.exception("AsyncHandle: %s raised", getattr(fn, "__name__", "fn"))
                return
            with handle._lock:
                if handle.state != "cancelled":
                    handle.result = value
                    handle.state = "done"

        handle._thread = threading.Thread(target=_runner, daemon=True, name="AsyncHandle")
        handle._thread.start()
        return handle

    def poll(self) -> State:
        """Return the current state. Non-blocking and idempotent."""
        return self.state

    def cancel(self) -> None:
        """Mark the handle cancelled. The thread keeps running (Python
        can't interrupt threads) but its result is discarded on completion."""
        with self._lock:
            if self.state in ("pending", "running"):
                self.state = "cancelled"

    def wait(self, timeout: Optional[float] = None) -> State:
        """Block until the background thread exits (test/synchronous use).
        Returns the final state."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.state


__all__ = ["AsyncHandle", "State"]
