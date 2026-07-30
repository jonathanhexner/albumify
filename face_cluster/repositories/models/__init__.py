"""ORM models for face_cluster repositories.

Importing this package registers all models with Base.metadata so that
alembic --autogenerate sees them. Each new model adds an import line here.
"""
from face_cluster.repositories.models.action_log import ActionLog

__all__ = ["ActionLog"]
