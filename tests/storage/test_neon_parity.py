"""spec-117 R3: the API, control and serving suites pass on Postgres (Neon) as well as SQLite.

``-m needs_neon`` only (excluded from the default run, so the suite stays SQLite, R15). The URL is read from
``.env`` (repo root, or ``ALBUMIFY_ENV_FILE``) inside the fixture and never printed. Each session gets its own
schema ``test_<random>``, dropped at the end; the child pytest reaches it through ``ALBUMIFY_TEST_DATABASE_URL``
(see ``tests/_db.py``). The schema is set per connection (libpq ``options``), which Neon's pooler does not pass
on, so the fixture uses the direct host.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from sim_bench.storage.database import DatabaseConfig, redact_url

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.needs_neon


def _neon_url() -> str | None:
    from dotenv import dotenv_values

    env_file = Path(os.environ.get("ALBUMIFY_ENV_FILE") or REPO / ".env")
    if not env_file.is_file():
        return None
    url = dotenv_values(env_file).get("DATABASE_URL")
    return url if url and not url.startswith("sqlite") else None


@pytest.fixture(scope="module")
def neon_schema():
    """(schema-scoped URL, secrets to scrub from any output). Dropped with everything in it afterwards."""
    url = _neon_url()
    if url is None:
        pytest.skip("needs_neon: no Postgres DATABASE_URL in .env")
    direct = url.replace("-pooler.", ".", 1)
    schema = f"test_{secrets.token_hex(4)}"
    admin = DatabaseConfig(direct).make_engine()
    try:
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        # application_name marks this run's connections: tests/_db.py may end only those
        scoped = make_url(direct).update_query_dict({"options": f"-csearch_path={schema}",
                                                     "application_name": f"albumify-test-{schema}"}) \
            .render_as_string(hide_password=False)
        scrub = [s for s in (url, direct, scoped, make_url(url).password) if s]
        yield scoped, schema, scrub
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


def _clean(out: str, scrub) -> str:
    for s in sorted(scrub, key=len, reverse=True):
        out = out.replace(s, redact_url(s) if "://" in s else "***")
    return out


def test_both_table_sets_live_in_one_postgres_schema(neon_schema):
    """R1 + R2 on Postgres: the API tables and the job tables side by side, ``people`` and ``job_people``."""
    url, schema, _ = neon_schema
    from sim_bench.api.database.models import Base
    from sim_bench.api.database.session import _migrate_new_columns
    from sim_bench.control.db import ControlDB

    engine = DatabaseConfig(url).make_engine()
    try:
        Base.metadata.create_all(engine)
        _migrate_new_columns(engine)
        db = ControlDB(url)
        names = set(inspect(engine).get_table_names())
        assert {"albums", "pipeline_runs", "people", "universal_cache", "jobs", "job_people", "photos"} <= names
        with engine.connect() as c:
            assert c.execute(text("select current_schema()")).scalar() == schema
        db.engine.dispose()
    finally:
        engine.dispose()


def test_api_control_serving_suites_pass_on_postgres(neon_schema):
    url, schema, scrub = neon_schema
    env = {**os.environ, "ALBUMIFY_TEST_DATABASE_URL": url}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/api", "tests/control", "tests/serving", "-q", "-p", "no:cacheprovider",
         "-W", "ignore::DeprecationWarning"],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3 * 3600)
    tail = _clean((proc.stdout or "")[-6000:] + (proc.stderr or "")[-2000:], scrub)
    assert proc.returncode == 0, tail
    assert " passed" in tail and " failed" not in tail, tail
    # the child really used Postgres: its tables are in this session's schema
    engine = DatabaseConfig(url).make_engine()
    try:
        assert {"jobs", "job_people", "pipeline_runs"} <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
