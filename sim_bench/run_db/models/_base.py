"""Declarative base for per-run face_clustering.db ORM models (spec-058).

This is a *separate* Base from ``face_cluster/repositories/_orm_base.py``
(sim_bench.db) on purpose — per-run DBs and the global sim_bench DB have
different lifecycles and different constraint-name namespaces. Never merge.
"""
from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


__all__ = ["Base", "NAMING_CONVENTION"]
