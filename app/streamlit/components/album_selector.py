"""Album selector and creation component."""

import streamlit as st
from pathlib import Path
from typing import Optional, Callable, List

from app.streamlit.api_client import get_client, ApiError
from app.streamlit.models import Album
from app.streamlit.session import get_session, set_current_album, add_notification


def render_album_selector(on_album_selected: Optional[Callable[[Album], None]] = None) -> Optional[Album]:
    """Render album selection dropdown."""
    state = get_session()
    client = get_client()
    albums = client.list_albums()

    if not albums:
        st.info("No albums yet. Create one below!")
        return None

    album_map = {f"{a.name} ({a.total_images} photos)": a for a in albums}
    options = ["Select an album..."] + list(album_map.keys())

    current_idx = 0
    if state.current_album:
        for i, (label, album) in enumerate(album_map.items(), 1):
            if album.album_id == state.current_album.album_id:
                current_idx = i
                break

    selected = st.selectbox("Album", options=options, index=current_idx, key="album_selector")

    if selected != "Select an album..." and selected in album_map:
        album = album_map[selected]
        if not state.current_album or album.album_id != state.current_album.album_id:
            set_current_album(album)
            if on_album_selected:
                on_album_selected(album)
        return album

    return state.current_album


def render_album_creator(on_created: Optional[Callable[[Album], None]] = None) -> None:
    """Render album creation form."""
    with st.expander("Create New Album", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            source_dir = st.text_input(
                "Source Directory",
                placeholder="C:/Photos/Vacation2024",
                key="new_album_source",
                help="Path to folder containing photos",
            )
            album_name = st.text_input(
                "Album Name",
                placeholder="Vacation 2024",
                key="new_album_name",
                help="A friendly name for this album",
            )

        with col2:
            st.write("")
            st.write("")
            if st.button("Create Album", type="primary", key="create_album_btn"):
                _create_album(source_dir, album_name, on_created)


def _create_album(
    source_dir: str,
    album_name: str,
    on_created: Optional[Callable[[Album], None]] = None
) -> None:
    """Create a new album."""
    if not source_dir:
        st.error("Please enter a source directory")
        return

    album_name = album_name or Path(source_dir).name
    source_path = Path(source_dir)

    if not source_path.exists() or not source_path.is_dir():
        st.error(f"Invalid directory: {source_dir}")
        return

    client = get_client()
    album = client.create_album(album_name, source_dir)
    add_notification(f"Created album: {album.name}", "success")
    set_current_album(album)
    if on_created:
        on_created(album)
    st.rerun()


def render_album_list(
    albums: List[Album],
    on_select: Optional[Callable[[Album], None]] = None,
    on_delete: Optional[Callable[[str], None]] = None,
) -> None:
    """Render a list of albums with actions."""
    state = get_session()

    for album in albums:
        is_current = state.current_album and album.album_id == state.current_album.album_id

        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                name_display = f"**{album.name}**" + (" ✓" if is_current else "")
                st.write(name_display)
                st.caption(album.source_directory)

            with col2:
                st.write(f"{album.total_images} photos")

            with col3:
                status = album.status.value if hasattr(album.status, 'value') else str(album.status)
                icons = {"idle": "🔘", "running": "🟡", "completed": "🟢", "failed": "🔴"}
                st.write(f"{icons.get(status, '⚪')} {status}")

            with col4:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📂", key=f"select_{album.album_id}", help="Select"):
                        set_current_album(album)
                        if on_select:
                            on_select(album)
                        st.rerun()
                with btn_col2:
                    if st.button("🗑", key=f"delete_{album.album_id}", help="Delete"):
                        if on_delete:
                            on_delete(album.album_id)

        st.divider()
