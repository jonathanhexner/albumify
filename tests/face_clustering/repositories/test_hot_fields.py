"""spec-048 Phase 3 — `ActionLog.hot_field_names()` matches the legacy tuple.

`_HOT_FIELDS` used to be a hand-maintained tuple in the Repository module.
It's now derived from per-column ``info={"updated_on_complete": True}``
metadata on the ORM model. This test pins both the contents and the
order, so a future column add/rename without the metadata flag is
caught immediately.
"""
from __future__ import annotations

from face_cluster.repositories.models.action_log import ActionLog

# The exact tuple the legacy _HOT_FIELDS held in spec-046. Order is the
# column declaration order in action_log.py.
_LEGACY_HOT_FIELDS: tuple[str, ...] = (
    "run_id", "source_dir", "output_dir", "album",
    "n_faces", "n_clusters", "n_noise", "log_file",
    "source_album", "run_name", "parent_run_id", "run_kind", "comment",
    "config_json", "n_core", "producer",
)


def test_hot_field_names_matches_legacy_tuple() -> None:
    assert ActionLog.hot_field_names() == _LEGACY_HOT_FIELDS


def test_hot_field_names_is_immutable_tuple() -> None:
    assert isinstance(ActionLog.hot_field_names(), tuple)


def test_hot_field_names_subset_of_columns() -> None:
    """Defensive: every hot field name actually exists on the model."""
    columns = set(ActionLog.__table__.columns.keys())
    assert set(ActionLog.hot_field_names()) <= columns
