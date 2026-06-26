"""spec-090: pick which pipeline run of an album to view.

Albumify used to always show the latest run (``results[0]``). This component lets
the user choose any run; the choice is stored in ``session_state.current_run_id``
and consumed by the Results / People pages.
"""

from typing import List, Dict, Optional

import streamlit as st

from app.streamlit.api_client import get_client
from app.streamlit.session import get_current_run_id, set_current_run_id


def _run_id(result: dict) -> str:
    return result.get("job_id") or result.get("id") or ""


def resolve_run_id(results: List[Dict], current: Optional[str]) -> Optional[str]:
    """Pick the run to show: the current selection if still valid, else the latest.

    ``results`` is the API list (newest first). Pure — unit-tested.
    """
    if not results:
        return None
    ids = [_run_id(r) for r in results]
    return current if current in ids else ids[0]


def render_run_selector(album_id: str) -> Optional[str]:
    """Render the Run dropdown for an album. Returns the selected run_id (or None)."""
    results = get_client().list_results(album_id)
    if not results:
        return None

    resolved = resolve_run_id(results, get_current_run_id())
    by_id = {_run_id(r): r for r in results}
    ids = list(by_id.keys())

    def _label(rid: str) -> str:
        r = by_id.get(rid, {})
        when = (r.get("completed_at") or r.get("created_at") or "")[:16].replace("T", " ")
        people = r.get("num_people")
        people_str = f"{people} people · " if people is not None else ""
        return f"{rid[:8]} · {when} · {people_str}{r.get('num_selected', '?')} selected"

    idx = ids.index(resolved) if resolved in ids else 0
    chosen = st.selectbox(
        "Run", ids, index=idx, format_func=_label, key=f"run_sel_{album_id}",
        help="Which pipeline run of this album to view.",
    )
    set_current_run_id(chosen)
    return chosen
