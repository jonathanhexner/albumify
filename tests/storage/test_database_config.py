"""spec-117 R1 / R15: one database setting; unset = SQLite exactly as before."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from sim_bench.storage.database import DatabaseConfig, load_env, redact_url

_PG = "postgresql://alice:s3cr3t@ep-x-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"


class ut_DatabaseConfig:
    def test_unset_is_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        cfg = DatabaseConfig.from_env(tmp_path / "sub" / "sim_bench.db")
        assert cfg.is_sqlite
        assert cfg.url == f"sqlite:///{(tmp_path / 'sub' / 'sim_bench.db').as_posix()}"
        assert (tmp_path / "sub").is_dir()

    def test_env_url_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABASE_URL", _PG)
        cfg = DatabaseConfig.from_env(tmp_path / "x.db")
        assert cfg.url == _PG and not cfg.is_sqlite

    def test_postgres_uses_psycopg3_with_pre_ping(self):
        cfg = DatabaseConfig(_PG)
        assert cfg.driver_url.startswith("postgresql+psycopg://alice:")
        assert DatabaseConfig("postgres://u:p@h/db").driver_url == "postgresql+psycopg://u:p@h/db"
        engine = cfg.make_engine()       # no connection is made until first use
        try:
            assert engine.dialect.name == "postgresql" and engine.dialect.driver == "psycopg"
            assert engine.pool._pre_ping is True
        finally:
            engine.dispose()

    def test_sqlite_engine_works_and_fk_only_when_asked(self, tmp_path):
        plain = DatabaseConfig.sqlite(tmp_path / "a.db").make_engine()
        fk = DatabaseConfig.sqlite(tmp_path / "b.db").make_engine(sqlite_foreign_keys=True)
        try:
            with plain.connect() as c:
                assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 0
            with fk.connect() as c:
                assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        finally:
            plain.dispose()
            fk.dispose()

    def test_redaction_hides_user_and_password(self):
        assert redact_url(_PG) == "postgresql://***@ep-x-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
        assert "s3cr3t" not in repr(DatabaseConfig(_PG))
        assert redact_url("sqlite:///C:/x.db") == "sqlite:///C:/x.db"


class ut_LoadEnv:
    def test_disabled_in_the_test_suite(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("ALBUMIFY_T117=1\n", encoding="utf-8")
        monkeypatch.delenv("ALBUMIFY_T117", raising=False)
        assert load_env(env) is False                      # conftest sets ALBUMIFY_NO_DOTENV=1
        import os
        assert "ALBUMIFY_T117" not in os.environ

    def test_reads_file_without_overriding(self, tmp_path, monkeypatch):
        import os
        env = tmp_path / ".env"
        env.write_text("ALBUMIFY_T117=from_file\nALBUMIFY_T117B=set\n", encoding="utf-8")
        monkeypatch.setenv("ALBUMIFY_NO_DOTENV", "0")
        monkeypatch.setenv("ALBUMIFY_T117", "already")
        monkeypatch.delenv("ALBUMIFY_T117B", raising=False)
        try:
            assert load_env(env) is True
            assert os.environ["ALBUMIFY_T117"] == "already" and os.environ["ALBUMIFY_T117B"] == "set"
        finally:
            os.environ.pop("ALBUMIFY_T117B", None)

    def test_missing_file_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALBUMIFY_NO_DOTENV", "0")
        assert load_env(tmp_path / "none.env") is False


def test_api_and_control_share_one_url_when_set(monkeypatch, tmp_path):
    """R1: with DATABASE_URL set, the API tables and the job tables land in the same database."""
    from sim_bench.api.database import session as api_session
    from sim_bench.control.db import ControlDB

    url = f"sqlite:///{(tmp_path / 'one.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(api_session, "_engine", None)
    monkeypatch.setattr(api_session, "_SessionLocal", None)
    assert api_session.get_database_url() == url
    api_session.init_db()
    ControlDB()
    try:
        names = set(inspect(api_session.get_engine()).get_table_names())
        assert {"albums", "pipeline_runs", "people", "universal_cache",
                "jobs", "photos", "faces", "job_people"} <= names
    finally:
        api_session.get_engine().dispose()


def test_unset_keeps_the_two_sqlite_files(monkeypatch, tmp_path):
    """R15: without DATABASE_URL the API stays on sim_bench.db and the jobs on control.db."""
    from sim_bench.api.database.session import get_database_url
    from sim_bench.control.db import default_url

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SIM_BENCH_DB", str(tmp_path / "sim_bench.db"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert get_database_url() == f"sqlite:///{(tmp_path / 'sim_bench.db').as_posix()}"
    assert default_url() == f"sqlite:///{(Path(tmp_path) / '.sim_bench' / 'control.db').as_posix()}"
