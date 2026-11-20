"""SIGHTING-152: GET/HEAD requests read in autocommit on Postgres (no BEGIN/ROLLBACK round trips); writes keep a
transaction. Checked on the factory choice, since the test database is SQLite (where both are the same session)."""
from types import SimpleNamespace

import pytest

from sim_bench.api.database import session as db_session
from sim_bench.control.db import ControlDB


@pytest.fixture
def two_factories(monkeypatch):
    made = []
    monkeypatch.setattr(db_session, "_SessionLocal", lambda: made.append("tx") or SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(db_session, "_ReadSessionLocal",
                        lambda: made.append("autocommit") or SimpleNamespace(close=lambda: None))
    return made


@pytest.mark.parametrize("method, kind", [("GET", "autocommit"), ("HEAD", "autocommit"), ("POST", "tx"),
                                          ("PATCH", "tx"), ("PUT", "tx"), ("DELETE", "tx")])
def test_only_read_requests_get_the_autocommit_session(two_factories, method, kind):
    gen = db_session.get_session(SimpleNamespace(method=method))
    next(gen)
    gen.close()
    assert two_factories == [kind]


def test_a_call_outside_a_request_keeps_the_transaction(two_factories):
    gen = db_session.get_session()       # API startup syncs the default profile through this
    next(gen)
    gen.close()
    assert two_factories == ["tx"]


def test_postgres_read_factories_are_autocommit_and_sqlite_shares_one(tmp_path, monkeypatch):
    import sqlalchemy
    # init_db on a Postgres URL, without a server: schema creation is stubbed, the engine never connects
    monkeypatch.setattr(db_session.Base.metadata, "create_all", lambda engine: None)
    monkeypatch.setattr(db_session, "_migrate_new_columns", lambda engine: None)
    for name in ("_engine", "_SessionLocal", "_ReadSessionLocal"):
        monkeypatch.setattr(db_session, name, None)
    db_session.init_db("postgresql://u:p@localhost/db")
    read_bind = db_session._ReadSessionLocal.kw["bind"]
    assert read_bind.get_execution_options().get("isolation_level") == "AUTOCOMMIT"
    assert "isolation_level" not in db_session._SessionLocal.kw["bind"].get_execution_options()

    control = ControlDB(f"sqlite:///{tmp_path / 'control.db'}")
    assert control._read_sessions is control._sessions
    with control.read_session() as s:
        assert s.execute(sqlalchemy.text("select 1")).scalar() == 1
