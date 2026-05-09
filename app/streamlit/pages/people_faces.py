"""People & Faces page - Merged people browsing and face management."""

import streamlit as st

from app.streamlit.session import get_session
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.api_client import get_client
from app.streamlit.components.people_browser import render_people_grid, render_person_detail, render_people_summary_row


def render_people_faces_page() -> None:
    """Render the merged People & Faces page."""
    st.header("People & Faces")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to view people and faces.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to view detected people.")
        return

    client = get_client()
    people = client.get_people(album.album_id)

    if not people:
        st.info("No people detected yet. Go to Configure & Run to run the pipeline with face clustering.")
        if st.button("Go to Configure & Run"):
            st.session_state.current_page = "configure"
            st.rerun()
        return

    # Check if viewing a specific person
    selected_person_id = st.session_state.get("selected_person_id")
    if selected_person_id:
        person = next((p for p in people if p.person_id == selected_person_id), None)
        if person:
            if st.button("< Back to all people"):
                st.session_state.selected_person_id = None
                st.rerun()
            render_person_detail(person, album.album_id, on_back=lambda: None, all_people=people)
            return

    # Summary metrics
    named = [p for p in people if p.name]
    unnamed = [p for p in people if not p.name]
    total_faces = sum(p.face_count for p in people)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("People", len(people))
    with col2:
        st.metric("Named", len(named))
    with col3:
        st.metric("Unnamed", len(unnamed))
    with col4:
        st.metric("Total Faces", total_faces)

    st.divider()

    # Named People section
    if named:
        st.subheader(f"Named People ({len(named)})")
        render_people_grid(named, columns=6, on_person_click=_on_person_click, album_id=album.album_id)
        st.divider()

    # Unnamed section
    if unnamed:
        st.subheader(f"Unnamed ({len(unnamed)})")
        st.caption("Click to name or merge these face groups")
        render_people_grid(unnamed, columns=6, on_person_click=_on_person_click, album_id=album.album_id)
        st.divider()

    # Needs Help section — placeholder for Phase D
    # TODO D1: Inline borderline faces from face_management.py
    st.subheader("Needs Help")
    st.caption("Borderline faces that need your decision")
    st.info("Face assistance coming in Phase D. Use the Face Management features in the meantime.")


def _on_person_click(person) -> None:
    """Handle person click - navigate to detail view."""
    st.session_state.selected_person_id = person.person_id
    st.rerun()
