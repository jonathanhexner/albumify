"""The one database setting for every Albumify table (spec-117 R1, R15).

    DATABASE_URL set    -> that database (Neon Postgres) holds the API tables, the job tables and the
                           feature cache
    DATABASE_URL unset  -> local SQLite files, exactly as before spec-117: the API and the feature cache in
                           ~/.sim_bench/sim_bench.db, the job tables in ~/.sim_bench/control.db

Nothing here reads ``.env`` on import. :func:`load_env` is called once by the processes that start work: the API
(``api/main.py`` startup) and the worker (``serving/runners.py`` child, ``serving/modal_app.py``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

__all__ = ["DatabaseConfig", "ENV_VAR", "NO_DOTENV_ENV", "ENV_FILE_ENV", "load_env", "redact_url"]

ENV_VAR = "DATABASE_URL"
NO_DOTENV_ENV = "ALBUMIFY_NO_DOTENV"      # "1": never read .env (the test suite sets it, R15)
ENV_FILE_ENV = "ALBUMIFY_ENV_FILE"        # another .env than the repo root's
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Optional[Path] = None) -> bool:
    """Read ``.env`` into the environment once at process start; variables already set win. True if a file was read."""
    if os.environ.get(NO_DOTENV_ENV) == "1":
        return False
    from dotenv import load_dotenv

    env_file = Path(path or os.environ.get(ENV_FILE_ENV) or _REPO_ROOT / ".env")
    if not env_file.is_file():
        return False
    load_dotenv(env_file, override=False)
    logger.info(".env loaded from %s (values not logged)", env_file)
    return True


def redact_url(url: str) -> str:
    """The URL with user and password replaced by ``***`` (scheme, host, database kept); safe for logs and error
    messages (R13)."""
    url = url or ""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """One database URL and how to connect to it."""

    url: str

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")

    @classmethod
    def from_env(cls, default_sqlite: Path) -> "DatabaseConfig":
        """``$DATABASE_URL``, else the SQLite file ``default_sqlite`` (its folder is created)."""
        url = os.environ.get(ENV_VAR)
        if url:
            return cls(url)
        return cls.sqlite(default_sqlite)

    @classmethod
    def sqlite(cls, path: Path) -> "DatabaseConfig":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{Path(path).as_posix()}")

    @property
    def driver_url(self) -> str:
        """Postgres URLs use psycopg 3 (``postgresql+psycopg://``); Neon hands out ``postgresql://``."""
        for prefix in ("postgres://", "postgresql://"):
            if self.url.startswith(prefix):
                return "postgresql+psycopg://" + self.url[len(prefix):]
        return self.url

    def make_engine(self, *, sqlite_timeout_s: Optional[float] = None, sqlite_foreign_keys: bool = False) -> Engine:
        """SQLite: one file, usable from several threads; foreign keys only when asked (the API's file never had
        them on). Postgres: pooled, every connection checked before use (Neon closes idle connections when it
        scales to zero)."""
        if self.is_sqlite:
            args = {"check_same_thread": False}
            if sqlite_timeout_s is not None:
                args["timeout"] = sqlite_timeout_s
            engine = create_engine(self.url, connect_args=args)
            if sqlite_foreign_keys:
                event.listen(engine, "connect", _sqlite_fk_on)
            return engine
        return create_engine(self.driver_url, pool_pre_ping=True, pool_recycle=300, pool_size=5, max_overflow=5)

    def redacted(self) -> str:
        return redact_url(self.url)

    def __repr__(self) -> str:                       # never the password, even in a traceback
        return f"DatabaseConfig({self.redacted()!r})"


def _sqlite_fk_on(dbapi_conn, _record) -> None:
    # SQLite ignores ON DELETE CASCADE unless foreign keys are switched on per connection.
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()
