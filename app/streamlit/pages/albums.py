"""Albums page (spec-116 R10): one table of runs, whichever machine ran them.

A row opens Results on that run; ticking two opens Compare runs. Real st.button rows, never dataframe row-select.
Albums themselves (create, select, delete) stay available below the table.
"""

from typing import Dict, List

import streamlit as st

from app.streamlit.session import get_session, clear_current_album, add_notification, set_current_album, \
    set_current_run_id
from app.streamlit.api_client import ApiError, get_client
from app.streamlit.models import Album
from app.streamlit.components.album_selector import render_album_creator, render_album_list

_WIDTHS = [0.5, 2, 3, 1.2, 1.6, 0.8, 0.8, 0.8, 1.1, 1]
_HEADERS = ["", "Album", "Folder", "Ran on", "Date", "Photos", "Picks", "People", "Status", ""]


def render_albums_page() -> None:
    """Render the albums page."""
    st.header("Albums")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to manage albums.")
        return

    render_album_creator(on_created=lambda a: add_notification(f"Created album: {a.name}", "success"))

    st.divider()
    _render_runs_table()

    st.divider()
    with st.expander("Manage albums"):
        _render_album_list()


def _render_runs_table() -> None:
    st.subheader("Runs")
    client = get_client()
    try:
        runs = client.list_runs()
    except ApiError as e:
        st.error(f"Cannot list runs: {e}")
        return
    if not runs:
        st.info("No runs yet. Start one from Configure & Run.")
        return

    for col, title in zip(st.columns(_WIDTHS), _HEADERS):
        col.markdown(f"**{title}**")
    focus = st.session_state.pop("albums_focus_run", None)
    ticked = [row for row in runs if _render_row(row, focus)]

    st.caption("Tick two finished runs to compare them. Runs from before the one run path have no job to compare.")
    if st.button("Compare runs", key="albums_compare", disabled=len(ticked) != 2):
        older, newer = sorted(ticked, key=lambda r: r["created_at"])
        _open_compare(older["run_id"], newer["run_id"])
        st.rerun()


def _render_row(row: Dict, focus) -> bool:
    """One run; returns whether its box is ticked."""
    run_id = row["run_id"]
    cols = st.columns(_WIDTHS)
    ticked = cols[0].checkbox("Compare", key=f"albums_tick_{run_id}", disabled=not row["comparable"],
                              label_visibility="collapsed")
    cols[1].write(f"**{row['album_name']}**" if run_id == focus else row["album_name"])
    cols[2].caption(row["folder"])
    cols[3].write(row["ran_on"])
    cols[4].write(str(row["created_at"])[:16].replace("T", " "))
    for col, name in zip(cols[5:8], ("photos", "picks", "people")):
        col.text("none" if row[name] is None else str(row[name]))    # st.write would make "-" a bullet
    cols[8].write(row["status"])
    if cols[9].button("Open", key=f"albums_open_{run_id}", disabled=row["status"] != "completed"):
        _open_run(row)
    return ticked


def _open_run(row: Dict) -> None:
    """Results on this run. The run dropdown keeps its own state, so it is pointed at the run too."""
    set_current_album(get_client().get_album(row["album_id"]))
    set_current_run_id(row["run_id"])
    st.session_state[f"run_sel_{row['album_id']}"] = row["run_id"]
    st.session_state["current_page"] = "results"
    st.rerun()


def _open_compare(reference: str, new: str) -> None:
    from app.streamlit.pages.compare import ANY
    st.session_state["compare_dataset"] = ANY
    st.session_state[f"compare_a_{ANY}"] = reference
    st.session_state[f"compare_b_{ANY}"] = new
    st.session_state["current_page"] = "compare"


def _render_album_list() -> None:
    """Render the list of albums."""
    client = get_client()
    albums = client.list_albums()

    if not albums:
        st.info("No albums yet. Create one above!")
        return

    # Summary stats
    total_albums = len(albums)
    total_photos = sum(a.total_images for a in albums)
    completed = sum(1 for a in albums if a.status.value == "completed")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Albums", total_albums)
    with col2:
        st.metric("Total Photos", total_photos)
    with col3:
        st.metric("Processed", f"{completed}/{total_albums}")

    st.divider()

    render_album_list(albums, on_select=_on_album_selected, on_delete=_on_album_delete)


def _on_album_selected(album: Album) -> None:
    """Handle album selection."""
    add_notification(f"Selected: {album.name}", "info")


def _on_album_delete(album_id: str) -> None:
    """Handle album deletion."""
    state = get_session()
    client = get_client()
    client.delete_album(album_id)
    add_notification("Album deleted", "success")

    if state.current_album and state.current_album.album_id == album_id:
        clear_current_album()

    st.rerun()
