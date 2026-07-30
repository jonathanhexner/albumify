"""spec-046 — replaces the four _COLUMNS drift-guard tests.

`alembic check` returns non-zero if Base.metadata diverges from the
head migration — i.e., if someone edits an ORM model without also
running `alembic revision --autogenerate`. This is the permanent
guard that SQLAlchemy + Alembic give us for free.

spec-048: ported off subprocess(alembic.exe) to the in-process
``alembic.command`` API. Captures Alembic's CommandError as the drift
signal (same exit-code-non-zero semantics, just via the Python API).
"""
from __future__ import annotations

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError

from face_cluster._paths import alembic_ini_path
from face_cluster.repositories._schema import ensure_schema


def test_orm_models_in_sync_with_alembic_head(tmp_path):
    """`alembic check` against a freshly-upgraded DB must report no drift."""
    db = tmp_path / "check.db"
    ensure_schema(db)

    cfg = Config(str(alembic_ini_path()))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    try:
        command.check(cfg)
    except CommandError as e:
        raise AssertionError(
            "`alembic check` reports drift between ORM models and head migration.\n"
            f"{e}\n"
            "Fix: run `.venv/Scripts/alembic revision --autogenerate -m \"<message>\"` "
            "to generate a migration capturing the model change."
        ) from e
