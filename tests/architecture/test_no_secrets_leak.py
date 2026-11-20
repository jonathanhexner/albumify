"""spec-117 R13: DATABASE_URL and R2_* live in .env (gitignored) and the Modal secret only -- never in logs, a command
line, a run record or the worker image. Uses made-up secrets; never reads the real .env."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from sim_bench.control.db import ControlDB
from sim_bench.storage.database import DatabaseConfig

REPO = Path(__file__).resolve().parents[2]
SECRET = "Sup3rS3cretPw"
PG = f"postgresql://owner:{SECRET}@ep-x-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"


def test_env_file_is_gitignored():
    out = subprocess.run(["git", "-C", str(REPO), "check-ignore", ".env"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ".env"


def test_database_logs_never_carry_the_password(caplog, monkeypatch, tmp_path):
    """Both engine factories log the URL redacted (the API's is pointed at a SQLite file so nothing connects)."""
    from sqlalchemy import create_engine

    from sim_bench.api.database import session as api_session

    caplog.set_level(logging.INFO)
    ControlDB(PG, create=False)                 # no connection is made
    monkeypatch.setattr(DatabaseConfig, "make_engine",
                        lambda self, **k: create_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}"))
    monkeypatch.setattr(api_session, "_engine", None)
    monkeypatch.setattr(api_session, "_SessionLocal", None)
    api_session.init_db(PG)
    text = caplog.text
    assert text.count("ep-x-pooler.us-east-1.aws.neon.tech") == 2
    assert SECRET not in text and "owner:" not in text
    assert SECRET not in repr(DatabaseConfig(PG))


def test_the_local_worker_gets_the_url_by_env_never_argv(tmp_path):
    from sim_bench.serving.job_store import JobStore
    from sim_bench.serving.runners import CONTROL_DB_ENV, LocalRunner

    runner = LocalRunner(JobStore(tmp_path), db=ControlDB(PG, create=False))
    assert SECRET not in " ".join(runner.command("j1"))
    assert runner.child_env()[CONTROL_DB_ENV] == PG


def test_a_run_record_holds_no_secret(tmp_path, monkeypatch):
    from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
    from sim_bench.serving.job_store import JobStore
    from tests.serving.test_run_output import RESULT, finished_context

    monkeypatch.setenv("DATABASE_URL", PG)
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", SECRET + "r2")
    store = JobStore(tmp_path / "jobs")
    store.inputs.put_bytes("00000000000000aa/a.jpg", b"x")
    store.write_input_manifest("j1", ["00000000000000aa/a.jpg"])
    JobCurator(store, runner=lambda *a, **k: (finished_context(), RESULT)).calc(
        CurateJobInputs(job_id="j1", runner="modal_gpu"))
    for f in store.paths("j1").root.rglob("*"):
        if f.is_file():
            assert SECRET not in f.read_text(encoding="utf-8", errors="replace"), f


def test_the_worker_image_gets_secrets_from_modal_only():
    """modal_app: the secret is mounted by name; no .env is added to the image; nothing sets the variables."""
    src = (REPO / "sim_bench" / "serving" / "modal_app.py").read_text(encoding="utf-8")
    assert "modal.Secret.from_name(CLOUD_SECRET)" in src
    assert ".env" not in src.replace("os.environ", "")
    assert "DATABASE_URL\"]" not in src and "R2_SECRET_ACCESS_KEY" not in src


def test_status_responses_carry_no_database_url():
    from sim_bench.api.schemas.pipeline import PipelineStatus
    fields = set(PipelineStatus.model_fields)
    assert not any("url" in f.lower() or "secret" in f.lower() for f in fields)
