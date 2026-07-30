"""Alembic migration environment for face_cluster.repositories.

Reads the target DB URL from env var SIM_BENCH_DB_URL (overrides any
value in alembic.ini). This lets tests point alembic at a temp DB
without editing the .ini file.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Importing the models package registers all ORM models with Base.metadata
# so that 'alembic --autogenerate' can detect them.
from face_cluster.repositories import models  # noqa: F401
from face_cluster.repositories._orm_base import Base


config = context.config

# When called from app code via alembic.command.* we don't want Alembic's
# fileConfig() to overwrite the host's root logger (it would discard the
# FileHandler installed by sim_bench.logging_setup, leaving fc_app_v2.log
# empty). Callers set ``cfg.attributes["configure_logger"] = False`` to opt
# out. The CLI continues to get logging via fileConfig.
if (
    config.config_file_name is not None
    and config.attributes.get("configure_logger", True)
):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    env_url = os.environ.get("SIM_BENCH_DB_URL")
    if env_url:
        return env_url
    ini_url = config.get_main_option("sqlalchemy.url")
    if not ini_url:
        raise RuntimeError(
            "No DB URL: set SIM_BENCH_DB_URL or sqlalchemy.url in alembic.ini"
        )
    return ini_url


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _db_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
