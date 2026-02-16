"""People page - Google Photos style people browser."""

import streamlit as st
from typing import List

from app.streamlit.session import get_session, add_notification
from app.streamlit.api_client import get_client
from app.streamlit.models import Person, Album
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.components.people_browser import render_people_grid, render_person_detail, render_merge_dialog
from app.streamlit.components.gallery import render_image_gallery


def render_people_page() -> None:
    """Render the people browser page."""
    st.header("People")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to browse people.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to browse people.")
        return

    st.divider()

    selected_person_id = st.session_state.get("selected_person_id")

    client = get_client()
    people = client.get_people(album.album_id)

    if not people:
        _render_no_people_state()
        return

    if selected_person_id:
        _render_person_detail_view(people, album, selected_person_id)
    else:
        _render_people_grid_view(people, album)


def _render_no_people_state() -> None:
    """Render state when no people detected."""
    st.info("No people detected in this album.")

    st.markdown("""
    **To detect people:**

    1. Go to **Results** page
    2. Run the **Full Pipeline** (with face detection)
    3. Come back here to browse people
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Results", type="primary"):
            st.session_state.current_page = "results"
            st.rerun()
    with col2:
        if st.button("Debug Face Analysis"):
            st.session_state.current_page = "debug"
            st.rerun()


def _render_people_grid_view(people: List[Person], album: Album) -> None:
    """Render the people grid overview."""
    total_people = len(people)
    named_count = sum(1 for p in people if p.name)
    total_faces = sum(p.face_count for p in people)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.metric("People Detected", total_people)
    with col2:
        st.metric("Named", f"{named_count}/{total_people}")
    with col3:
        st.metric("Total Faces", total_faces)
    with col4:
        manage_mode = st.checkbox("Manage", value=st.session_state.get("manage_mode", False), key="manage_mode_toggle", help="Enable to merge multiple people")
        st.session_state.manage_mode = manage_mode

    # Debug summary expander
    with st.expander("Face Filtering Summary", expanded=False):
        st.markdown("""
        **How faces are filtered for clustering:**

        1. **Basic filters** (`filter_faces` step): Remove small/low-confidence faces
           - Confidence >= 0.5, BBox ratio >= 0.02, Relative size >= 0.3

        2. **Frontal filters** (`score_face_frontal` step): Mark non-frontal as non-clusterable
           - Frontal score >= 0.4 (from eye/bbox ratio + asymmetry)

        3. **Embedding extraction**: Only clusterable faces get embeddings

        4. **Clustering**: Groups faces by embedding similarity (HDBSCAN)
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Clustered faces**: {total_faces}")
        with col2:
            if st.button("Open Debug Page", key="open_debug_from_people"):
                st.session_state.current_page = "debug"
                st.rerun()

    st.divider()

    if st.session_state.get("show_merge_dialog"):
        render_merge_dialog(people, album.album_id, on_complete=lambda: st.session_state.update(show_merge_dialog=False))
        st.divider()

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        search = st.text_input("Search by name", placeholder="Type to filter...", key="people_search", label_visibility="collapsed")

    with col2:
        sort_by = st.selectbox("Sort by", ["Most photos", "Name", "Recently added"], key="people_sort", label_visibility="collapsed")

    with col3:
        if manage_mode and st.button("Open Merge Dialog", key="open_merge"):
            st.session_state.show_merge_dialog = True
            st.rerun()

    filtered_people = people
    if search:
        search_lower = search.lower()
        filtered_people = [p for p in people if (p.name and search_lower in p.name.lower()) or search_lower in p.person_id.lower()]

    if sort_by == "Most photos":
        filtered_people = sorted(filtered_people, key=lambda p: p.face_count, reverse=True)
    elif sort_by == "Name":
        filtered_people = sorted(filtered_people, key=lambda p: (p.name or "zzz").lower())

    st.write(f"Showing **{len(filtered_people)}** of {len(people)} people")

    def on_person_click(person: Person):
        st.session_state.selected_person_id = person.person_id
        st.rerun()

    render_people_grid(filtered_people, columns=5, on_person_click=on_person_click, enable_selection=manage_mode, album_id=album.album_id)


def _render_person_detail_view(people: List[Person], album: Album, person_id: str) -> None:
    """Render detailed view for a specific person."""
    person = next((p for p in people if p.person_id == person_id), None)

    if not person:
        st.error("Person not found")
        _go_back_to_grid()
        return

    render_person_detail(person, album.album_id, on_back=_go_back_to_grid, all_people=people)

    st.divider()
    _render_related_people(people, person)


def _render_related_people(all_people: List[Person], current_person: Person) -> None:
    """Render section showing related people."""
    with st.expander("Related People", expanded=False):
        st.caption("People who often appear in the same photos")

        other_people = [p for p in all_people if p.person_id != current_person.person_id][:4]

        if not other_people:
            st.info("No other people detected")
            return

        cols = st.columns(len(other_people))

        for i, person in enumerate(other_people):
            with cols[i]:
                display_name = person.name or f"Person {i+1}"
                st.write(f"**{display_name}**")
                st.caption(f"{person.face_count} photos")

                if st.button("View", key=f"related_{person.person_id}"):
                    st.session_state.selected_person_id = person.person_id
                    st.rerun()


def _go_back_to_grid() -> None:
    """Clear selection and go back to grid view."""
    st.session_state.selected_person_id = None
    st.session_state.show_rename_dialog = False
    st.rerun()
