"""RunMetadataRow ORM model — mirrors RUN_METADATA_DDL.

Class suffix "Row" avoids name clash with the ``RunMetadata`` dataclass
in ``face_cluster.run_history``.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, TEXT
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class RunMetadataRow(Base):
    __tablename__ = "run_metadata"

    run_id: Mapped[str] = mapped_column(TEXT, primary_key=True)
    source_album: Mapped[str] = mapped_column(TEXT, nullable=False)
    producer: Mapped[str] = mapped_column(TEXT, nullable=False)
    parent_run_id: Mapped[Optional[str]] = mapped_column(TEXT)
    config_json: Mapped[str] = mapped_column(TEXT, nullable=False)
    merge_thresholds_json: Mapped[Optional[str]] = mapped_column(TEXT)
    merge_iter_summary_json: Mapped[Optional[str]] = mapped_column(TEXT)
    n_images: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_faces: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_core: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_clusters_base: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_clusters_final: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_merges: Mapped[int] = mapped_column(INTEGER, nullable=False)
    n_iterations: Mapped[int] = mapped_column(INTEGER, nullable=False)
    started_at: Mapped[str] = mapped_column(TEXT, nullable=False)
    finished_at: Mapped[str] = mapped_column(TEXT, nullable=False)
    schema_version: Mapped[int] = mapped_column(INTEGER, nullable=False)
