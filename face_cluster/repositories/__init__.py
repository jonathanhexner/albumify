"""Repository layer for face_cluster persistence.

Each ``*Repository`` class owns one or more storage backends (SQLite
tables, files) and exposes typed query / mutation methods. Services
in ``face_cluster.views`` compose Repositories via constructor
injection; they never bypass.

Pattern reference: ``docs/architecture/architecture_standards.md``
§B0 (Repository pattern), §A6 (per-tab migration discipline).
"""
from face_cluster.repositories._errors import (
    NotFoundError,
    RepositoryError,
    ValidationError,
)
from face_cluster.repositories.run_history_repo import (
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
)

__all__ = [
    "NotFoundError",
    "RepositoryError",
    "RunHistoryCriteria",
    "RunHistoryRepoConfig",
    "RunHistoryRepository",
    "ValidationError",
]
