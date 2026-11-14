"""SIGHTING-156 (spec-117 review H1): a lost connection to Modal is "unknown", never "failed".

The watchdog tests use a fake remote; these drive the REAL ``DetachedModalRemote.status`` with the real exception
types Modal and grpclib raise, so the gap the fake hid (the fake raised where the real code returned "failed")
cannot come back."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

# Importing modal switches asyncio to the Selector loop on Windows, which breaks every later Playwright test
# (subprocesses need the Proactor loop). Keep the policy as it was before this file was collected.
_POLICY = asyncio.get_event_loop_policy()
import modal  # noqa: E402
asyncio.set_event_loop_policy(_POLICY)
from grpclib.const import Status
from grpclib.exceptions import GRPCError, StreamTerminatedError
from modal.exception import ConnectionError as ModalConnectionError
from modal.exception import FunctionTimeoutError, OutputExpiredError, RemoteError

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, RUNNING, STAGING, JobRepo
from sim_bench.serving.detached_modal import DONE, GONE, UNKNOWN, DetachedModalRemote, call_ended
from sim_bench.serving.run_watchdog import RunWatchdog

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


class _Call:
    def __init__(self, outcome):
        self.outcome = outcome

    def get(self, timeout=None):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.fixture(autouse=True)
def _keep_event_loop_policy():
    yield
    asyncio.set_event_loop_policy(_POLICY)


@pytest.fixture
def modal_call(monkeypatch):
    """Make ``modal.FunctionCall.from_id(...).get()`` return or raise ``outcome``."""
    def _set(outcome):
        monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda call_id: _Call(outcome)))
    return _set


LOST_CONNECTION = [
    ModalConnectionError("could not connect to api.modal.com"),
    OSError("network is unreachable"),
    ConnectionResetError("connection reset by peer"),
    StreamTerminatedError("stream reset"),
    GRPCError(Status.UNAVAILABLE, "upstream connect error"),
    GRPCError(Status.DEADLINE_EXCEEDED, "deadline exceeded"),
]


class ut_DetachedModalStatus:
    @pytest.mark.parametrize("err", LOST_CONNECTION, ids=lambda e: type(e).__name__)
    def test_a_lost_connection_is_unknown_not_failed(self, modal_call, err):
        modal_call(err)
        status = DetachedModalRemote().status("call-1")
        assert status == UNKNOWN
        assert not call_ended(status)

    @pytest.mark.parametrize("err", [FunctionTimeoutError("timed out"), RemoteError("crashed"),
                                     GRPCError(Status.NOT_FOUND, "no such call")],
                             ids=lambda e: type(e).__name__)
    def test_a_real_failure_of_the_call_is_still_failed(self, modal_call, err):
        modal_call(err)
        status = DetachedModalRemote().status("call-1")
        assert status.startswith("failed: ") and call_ended(status)

    def test_running_done_and_gone_unchanged(self, modal_call):
        modal_call(TimeoutError())
        assert DetachedModalRemote().status("c") == "running"
        modal_call(None)
        assert DetachedModalRemote().status("c") == DONE
        modal_call(OutputExpiredError())
        assert DetachedModalRemote().status("c") == GONE


class ut_WatchdogWithRealRemote:
    """The H1 scenario end to end: fresh heartbeat, PC loses Modal, the watchdog runs."""

    @pytest.fixture
    def repo(self, control_db_url):
        db = ControlDB(control_db_url)
        repo = JobRepo(db)
        repo.create("j1", runner="modal_gpu")
        repo.transition("j1", STAGING, worker="pc:1:1")
        repo.transition("j1", RUNNING)
        repo.set_progress("j1", {"stage": "running"}, heartbeat=False, call_id="call-1",
                          heartbeat_at=NOW - timedelta(seconds=10))
        repo.db = db
        return repo

    def test_a_network_blip_does_not_fail_a_healthy_run(self, repo, modal_call):
        modal_call(ModalConnectionError("offline"))
        dog = RunWatchdog(repo.db, DetachedModalRemote(), grace_s=90, clock=lambda: NOW)
        assert dog.reconcile() == 0
        assert repo.get("j1").status == RUNNING

    def test_a_crashed_call_still_fails(self, repo, modal_call):
        modal_call(RemoteError("worker crashed"))
        dog = RunWatchdog(repo.db, DetachedModalRemote(), grace_s=90, clock=lambda: NOW)
        assert dog.reconcile() == 1
        assert repo.get("j1").status == FAILED
