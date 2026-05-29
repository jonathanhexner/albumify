"""spec-050 / spec-051 — recent-runs picker component.

Renders a single ``st.selectbox`` listing the most recent v2 runs from
the global action_log. Sourced from ``RunHistoryRepository`` — no disk
scan. The History tab and this picker therefore agree on which runs
exist; the v2 app has one source of truth.

spec-051: rows whose ``output_dir / face_clustering.db`` no longer
exists on disk are kept visible but flagged with a ``[missing]``
prefix; the Clusters tab refuses to load them and shows a warning.
A footnote below the dropdown summarises the orphan count so the user
knows the cleanup script (``scripts/cleanup_orphan_action_log.py``) is
worth running.

Default-selects the first non-orphan entry, falling back to the entry
matching ``st.session_state.v2_last_run_dir`` when present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from face_cluster.repositories import (
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
)


@dataclass(frozen=True, slots=True)
class RunPickerEntry:
    """One row in the run picker — everything the Clusters tab needs to
    load and label a run, projected from the action_log row.

    ``is_orphan`` means the recorded ``output_dir`` no longer contains a
    ``face_clustering.db`` (directory was deleted, moved, or the row was
    pollution from a now-cleaned test fixture).
    """
    run_id: str
    output_dir: Path
    album: str
    started_at: str
    n_faces: int
    n_clusters: int
    status: str  # 'complete' | 'failed' | 'running'
    is_orphan: bool = False


def _all_entries_from_repo(
    *, limit: int, db_path: Optional[Path] = None,
) -> List[RunPickerEntry]:
    """Fetch the ``limit`` most recent v2 runs, newest first.

    Includes orphan rows (whose output_dir/face_clustering.db is gone).
    Call ``_partition_entries`` to split loadable from orphan, or read
    ``entry.is_orphan`` directly.
    """
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path))
    rows = repo.find(RunHistoryCriteria(producer="fc_app_v2", limit=limit))
    entries: List[RunPickerEntry] = []
    for r in rows:
        if not r.output_dir or not r.run_id:
            # Rows without an output_dir aren't loadable; skip entirely.
            continue
        out_dir = Path(r.output_dir)
        is_orphan = not (out_dir / "face_clustering.db").exists()
        entries.append(RunPickerEntry(
            run_id=r.run_id,
            output_dir=out_dir,
            album=r.source_album or "(unknown)",
            started_at=r.started_at or "",
            n_faces=int(r.n_faces or 0),
            n_clusters=int(r.n_clusters or 0),
            status=r.status or "",
            is_orphan=is_orphan,
        ))
    return entries


# Backwards-compat alias for spec-050 tests that still expect the old name.
_entries_from_repo = _all_entries_from_repo


def _partition_entries(
    entries: List[RunPickerEntry],
) -> Tuple[List[RunPickerEntry], List[RunPickerEntry]]:
    """Return (loadable, orphan) — orphans go AFTER loadable in the dropdown."""
    loadable = [e for e in entries if not e.is_orphan]
    orphan = [e for e in entries if e.is_orphan]
    return loadable, orphan


def _format_label(e: RunPickerEntry) -> str:
    when = e.started_at[:16].replace("T", " ") if e.started_at else "?"
    counts = f"{e.n_faces or '-'}f/{e.n_clusters or '-'}c"
    prefix = "[missing] " if e.is_orphan else ""
    return f"{prefix}{when}  ·  {e.album}  ·  {counts}  ·  {e.status}  ({e.run_id[:8]})"


def render_run_picker(
    *,
    label: str = "Pick a run",
    key: str = "v2_run_picker",
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> Optional[RunPickerEntry]:
    """Render the selectbox. Returns the selected ``RunPickerEntry`` or
    ``None`` when no v2 runs exist yet."""
    all_entries = _all_entries_from_repo(limit=limit, db_path=db_path)
    if not all_entries:
        st.info("No v2 runs in action_log yet. Run something from the Run tab.")
        return None

    loadable, orphan = _partition_entries(all_entries)
    ordered = loadable + orphan  # loadable first; orphans visible but ranked last

    # Default to the entry that matches the last Run's output_dir, else the
    # first non-orphan entry, else index 0.
    last_dir = st.session_state.get("v2_last_run_dir", "")
    default_index = 0
    for i, e in enumerate(ordered):
        if str(e.output_dir) == last_dir:
            default_index = i
            break
    else:
        # No match — prefer the first loadable entry over an orphan.
        if loadable:
            default_index = 0  # loadable[0] is already at index 0 of ordered

    options = list(range(len(ordered)))
    chosen_index = st.selectbox(
        label,
        options=options,
        index=default_index,
        format_func=lambda i: _format_label(ordered[i]),
        key=key,
    )

    if orphan:
        st.caption(
            f"⚠ {len(orphan)} orphan run(s) shown with `[missing]` — their "
            f"directory no longer exists. Run "
            f"`scripts/cleanup_orphan_action_log.py --dry-run` to inspect, "
            f"then `--apply` to remove."
        )

    return ordered[chosen_index]


__all__ = ["RunPickerEntry", "render_run_picker"]
