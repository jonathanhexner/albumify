"""Session factory + session_scope context manager.

Usage:
    sm = make_sessionmaker(engine)
    with session_scope(sm) as session:
        repo = RunHistoryRepository(session)
        ...
    # commit happened on clean exit; rollback happened on exception.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(sm: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = sm()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["make_sessionmaker", "session_scope"]
