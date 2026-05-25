"""Repository-layer error hierarchy.

A small typed hierarchy so callers can distinguish failure modes
without string-matching messages. Future merger with the Service-layer
hierarchy in ``face_cluster.views._errors`` (spec-042 B2 follow-up)
will collapse both into a single ``ServiceError`` family — for now,
this module is the seed.

Conventions:
* All Repository methods raise ``RepositoryError`` (or a subclass);
  never plain ``ValueError`` / ``RuntimeError``.
* Each subclass carries an optional ``user_message`` safe to display
  to end users; the base ``args[0]`` carries the developer-facing
  detail.
"""
from __future__ import annotations

from typing import Optional


class RepositoryError(Exception):
    """Base for all Repository-layer errors.

    Carries an optional ``user_message`` — a short, end-user-safe
    string the UI layer can render directly. If not provided, falls
    back to the developer message in ``args[0]``.
    """

    def __init__(self, message: str, *, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or message


class NotFoundError(RepositoryError):
    """Requested entity does not exist.

    Raised when a method targeting a specific id (e.g.,
    ``RunHistoryRepository.complete_action(action_id=...)``) finds no
    matching row.
    """


class ValidationError(RepositoryError):
    """Inputs do not satisfy the contract.

    Raised when arguments to a mutation method fail validation —
    e.g., a comment exceeding the maximum length, a non-positive id,
    a date range that's inverted.
    """


__all__ = ["RepositoryError", "NotFoundError", "ValidationError"]
