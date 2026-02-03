"""Albums page - Album management."""

import streamlit as st

from app.streamlit.session import get_session, clear_current_album, add_notification
from app.streamlit.api_client import get_client
from app.streamlit.models import Album
from app.streamlit.components.album_selector import render_album_creator, render_album_list


def render_albums_page() -> None:
    """Render the albums management page."""
    st.header("Albums")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to manage albums.")
        return

    render_album_creator(on_created=lambda a: add_notification(f"Created album: {a.name}", "success"))

    st.divider()
    _render_album_list()


def _render_album_list() -> None:
    """Render the list of albums."""
    st.subheader("Your Albums")

    state = get_session()
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
