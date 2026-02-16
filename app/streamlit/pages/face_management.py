"""Face Management page - Correct face clustering assignments."""

import streamlit as st
from typing import List, Optional

from app.streamlit.session import get_session, add_notification
from app.streamlit.api_client import get_client
from app.streamlit.models import (
    FaceInfo,
    BorderlineFace,
    PersonSummary,
    FaceAction,
    BatchChangeResponse,
)
from app.streamlit.components.album_selector import render_album_selector


def render_face_management_page() -> None:
    """Render the face management page."""
    st.header("Face Management")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to manage faces.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to manage faces.")
        return

    # Get active run ID
    run_id = _get_active_run_id(album.album_id)
    if not run_id:
        _render_no_run_state()
        return

    st.divider()

    # Initialize session state for face management
    _init_face_management_state()

    # Render tabs for different views
    tab_needs_help, tab_all_faces, tab_people, tab_pending = st.tabs([
        "Needs Help",
        "All Faces",
        "People",
        "Pending Changes"
    ])

    with tab_needs_help:
        _render_needs_help_tab(album.album_id, run_id)

    with tab_all_faces:
        _render_all_faces_tab(album.album_id, run_id)

    with tab_people:
        _render_people_tab(album.album_id, run_id)

    with tab_pending:
        _render_pending_changes_tab(album.album_id, run_id)


def _get_active_run_id(album_id: str) -> Optional[str]:
    """Get the active pipeline run ID for an album."""
    client = get_client()
    results = client.list_results(album_id)
    if results:
        # Return the most recent result's job_id
        latest = results[0]
        return latest.get("job_id", latest.get("id", ""))
    return None


def _render_no_run_state() -> None:
    """Render state when no pipeline run exists."""
    st.info("No pipeline run found for this album.")

    st.markdown("""
    **To manage faces:**

    1. Go to **Results** page
    2. Run the **Full Pipeline** with face detection
    3. Come back here to manage face assignments
    """)

    if st.button("Go to Results", type="primary"):
        st.session_state.current_page = "results"
        st.rerun()


def _init_face_management_state() -> None:
    """Initialize session state for face management."""
    if "fm_pending_changes" not in st.session_state:
        st.session_state.fm_pending_changes = []
    if "fm_mode" not in st.session_state:
        st.session_state.fm_mode = "batch"  # batch or live
    if "fm_selected_faces" not in st.session_state:
        st.session_state.fm_selected_faces = set()


def _render_needs_help_tab(album_id: str, run_id: str) -> None:
    """Render the Needs Help wizard tab."""
    st.subheader("Faces Needing Your Decision")

    st.markdown("""
    These faces are borderline - the system isn't confident enough to auto-assign them.
    Help improve accuracy by confirming or rejecting assignments.
    """)

    client = get_client()
    borderline_faces = client.get_borderline_faces(album_id, run_id, limit=10)

    if not borderline_faces:
        st.success("No faces need your help right now!")
        return

    st.write(f"**{len(borderline_faces)}** faces need decisions")

    for i, bf in enumerate(borderline_faces):
        _render_borderline_face_card(album_id, run_id, bf, i)


def _render_borderline_face_card(
    album_id: str,
    run_id: str,
    borderline: BorderlineFace,
    index: int
) -> None:
    """Render a single borderline face decision card."""
    face = borderline.face

    with st.container():
        col_face, col_person, col_actions = st.columns([1, 1, 2])

        with col_face:
            st.write("**This face:**")
            if face.thumbnail_base64:
                st.image(
                    f"data:image/jpeg;base64,{face.thumbnail_base64}",
                    width=100
                )
            else:
                st.write("(No thumbnail)")

        with col_person:
            st.write(f"**Closest match: {borderline.closest_person_name}**")
            if borderline.closest_person_thumbnail:
                st.image(
                    f"data:image/jpeg;base64,{borderline.closest_person_thumbnail}",
                    width=100
                )
            st.caption(f"Distance: {borderline.distance:.3f}")
            st.caption(f"Uncertainty: {borderline.uncertainty_score:.2f}")

        with col_actions:
            st.write("**Decision:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    f"Yes, this is {borderline.closest_person_name}",
                    key=f"confirm_{index}",
                    type="primary"
                ):
                    _add_pending_change(FaceAction(
                        face_key=face.face_key,
                        action="assign",
                        target_person_id=borderline.closest_person_id
                    ))
                    add_notification(f"Added assignment to {borderline.closest_person_name}", "success")
                    st.rerun()

            with col2:
                if st.button("No, someone else", key=f"other_{index}"):
                    st.session_state[f"show_assign_menu_{index}"] = True
                    st.rerun()

            with col3:
                if st.button("Skip/Untag", key=f"skip_{index}"):
                    _add_pending_change(FaceAction(
                        face_key=face.face_key,
                        action="untag"
                    ))
                    add_notification("Face will be untagged", "info")
                    st.rerun()

            # Show assign menu if requested
            if st.session_state.get(f"show_assign_menu_{index}"):
                _render_assign_menu(album_id, run_id, face, index)

        st.divider()


def _render_assign_menu(
    album_id: str,
    run_id: str,
    face: FaceInfo,
    index: int
) -> None:
    """Render the person assignment menu."""
    client = get_client()
    distances = client.get_face_distances(album_id, run_id, face.face_key)

    st.write("**Assign to:**")

    for dist in distances[:5]:  # Show top 5 closest
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{dist.person_name} (distance: {dist.centroid_distance:.3f})")
        with col2:
            if st.button("Select", key=f"assign_{index}_{dist.person_id}"):
                _add_pending_change(FaceAction(
                    face_key=face.face_key,
                    action="assign",
                    target_person_id=dist.person_id
                ))
                st.session_state[f"show_assign_menu_{index}"] = False
                add_notification(f"Added assignment to {dist.person_name}", "success")
                st.rerun()

    # Option to create new person
    new_name = st.text_input("Or create new person:", key=f"new_person_{index}")
    if new_name and st.button("Create & Assign", key=f"create_{index}"):
        _add_pending_change(FaceAction(
            face_key=face.face_key,
            action="assign",
            new_person_name=new_name
        ))
        st.session_state[f"show_assign_menu_{index}"] = False
        add_notification(f"Will create new person: {new_name}", "success")
        st.rerun()


def _render_all_faces_tab(album_id: str, run_id: str) -> None:
    """Render tab showing all faces with filtering."""
    st.subheader("All Faces")

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        status_filter = st.multiselect(
            "Filter by status",
            options=["assigned", "unassigned", "untagged", "not_a_face"],
            default=["assigned", "unassigned"],
            key="face_status_filter"
        )

    with col2:
        view_mode = st.radio(
            "View",
            options=["Grid", "List"],
            horizontal=True,
            key="face_view_mode"
        )

    with col3:
        st.write("")  # Spacer
        if st.button("Refresh", key="refresh_faces"):
            st.rerun()

    # Load faces
    client = get_client()
    faces = client.get_faces(album_id, run_id, status=status_filter if status_filter else None)

    if not faces:
        st.info("No faces match the current filter.")
        return

    st.write(f"Showing **{len(faces)}** faces")

    if view_mode == "Grid":
        _render_faces_grid(album_id, run_id, faces)
    else:
        _render_faces_list(album_id, run_id, faces)


def _render_faces_grid(album_id: str, run_id: str, faces: List[FaceInfo]) -> None:
    """Render faces in a grid layout."""
    cols_per_row = 5
    rows = [faces[i:i + cols_per_row] for i in range(0, len(faces), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for i, face in enumerate(row):
            with cols[i]:
                _render_face_card(album_id, run_id, face)


def _render_face_card(album_id: str, run_id: str, face: FaceInfo) -> None:
    """Render a single face card in grid view."""
    # Thumbnail
    if face.thumbnail_base64:
        st.image(
            f"data:image/jpeg;base64,{face.thumbnail_base64}",
            use_container_width=True
        )
    else:
        st.write("(No image)")

    # Status badge
    status_colors = {
        "assigned": "green",
        "unassigned": "orange",
        "untagged": "gray",
        "not_a_face": "red"
    }
    color = status_colors.get(face.status, "gray")
    st.markdown(f":{color}[{face.status}]")

    # Person name if assigned
    if face.person_name:
        st.caption(face.person_name)

    # Selection checkbox
    is_selected = face.face_key in st.session_state.fm_selected_faces
    if st.checkbox("Select", value=is_selected, key=f"sel_{face.face_key}"):
        st.session_state.fm_selected_faces.add(face.face_key)
    else:
        st.session_state.fm_selected_faces.discard(face.face_key)


def _render_faces_list(album_id: str, run_id: str, faces: List[FaceInfo]) -> None:
    """Render faces in a list layout with more details."""
    for face in faces:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

            with col1:
                if face.thumbnail_base64:
                    st.image(
                        f"data:image/jpeg;base64,{face.thumbnail_base64}",
                        width=80
                    )

            with col2:
                st.write(f"**{face.face_key}**")
                st.caption(f"Status: {face.status}")
                if face.person_name:
                    st.caption(f"Person: {face.person_name}")

            with col3:
                if face.frontal_score is not None:
                    st.caption(f"Frontal: {face.frontal_score:.2f}")
                if face.centroid_distance is not None:
                    st.caption(f"Distance: {face.centroid_distance:.3f}")
                if face.assignment_method:
                    st.caption(f"Method: {face.assignment_method}")

            with col4:
                if st.button("Actions", key=f"actions_{face.face_key}"):
                    st.session_state[f"show_face_actions_{face.face_key}"] = True
                    st.rerun()

            # Show action menu if requested
            if st.session_state.get(f"show_face_actions_{face.face_key}"):
                _render_face_action_menu(album_id, run_id, face)

            st.divider()


def _render_face_action_menu(album_id: str, run_id: str, face: FaceInfo) -> None:
    """Render action menu for a face."""
    st.write("**Actions:**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Assign", key=f"menu_assign_{face.face_key}"):
            st.session_state[f"show_assign_to_{face.face_key}"] = True
            st.rerun()

    with col2:
        if face.status == "assigned":
            if st.button("Remove", key=f"menu_remove_{face.face_key}"):
                _add_pending_change(FaceAction(
                    face_key=face.face_key,
                    action="unassign"
                ))
                st.session_state[f"show_face_actions_{face.face_key}"] = False
                add_notification("Face will be removed from person", "info")
                st.rerun()

    with col3:
        if st.button("Untag", key=f"menu_untag_{face.face_key}"):
            _add_pending_change(FaceAction(
                face_key=face.face_key,
                action="untag"
            ))
            st.session_state[f"show_face_actions_{face.face_key}"] = False
            add_notification("Face will be untagged", "info")
            st.rerun()

    with col4:
        if st.button("Not a Face", key=f"menu_notface_{face.face_key}"):
            _add_pending_change(FaceAction(
                face_key=face.face_key,
                action="not_a_face"
            ))
            st.session_state[f"show_face_actions_{face.face_key}"] = False
            add_notification("Marked as not a face", "info")
            st.rerun()

    # Show assign dialog
    if st.session_state.get(f"show_assign_to_{face.face_key}"):
        client = get_client()
        people = client.get_people_summary(album_id, run_id)

        selected_person = st.selectbox(
            "Assign to:",
            options=[p.person_id for p in people],
            format_func=lambda pid: next((p.name for p in people if p.person_id == pid), pid),
            key=f"assign_select_{face.face_key}"
        )

        if st.button("Confirm", key=f"confirm_assign_{face.face_key}"):
            _add_pending_change(FaceAction(
                face_key=face.face_key,
                action="assign",
                target_person_id=selected_person
            ))
            st.session_state[f"show_assign_to_{face.face_key}"] = False
            st.session_state[f"show_face_actions_{face.face_key}"] = False
            add_notification("Assignment added to pending changes", "success")
            st.rerun()

    if st.button("Close", key=f"close_actions_{face.face_key}"):
        st.session_state[f"show_face_actions_{face.face_key}"] = False
        st.rerun()


def _render_people_tab(album_id: str, run_id: str) -> None:
    """Render tab showing people with their exemplars."""
    st.subheader("People Overview")

    client = get_client()
    people = client.get_people_summary(album_id, run_id)

    if not people:
        st.info("No people found.")
        return

    st.write(f"**{len(people)}** people detected")

    for person in people:
        with st.expander(f"{person.name} ({person.face_count} faces)", expanded=False):
            col1, col2 = st.columns([1, 3])

            with col1:
                if person.thumbnail_base64:
                    st.image(
                        f"data:image/jpeg;base64,{person.thumbnail_base64}",
                        width=100
                    )
                st.caption(f"ID: {person.person_id[:8]}...")

            with col2:
                st.write(f"**Face count:** {person.face_count}")
                st.write(f"**Exemplars:** {person.exemplar_count}")
                if person.cluster_tightness is not None:
                    st.write(f"**Cluster tightness:** {person.cluster_tightness:.3f}")

                # Actions
                if st.button("Rename", key=f"rename_{person.person_id}"):
                    st.session_state[f"show_rename_{person.person_id}"] = True
                    st.rerun()

            # Rename dialog
            if st.session_state.get(f"show_rename_{person.person_id}"):
                new_name = st.text_input(
                    "New name:",
                    value=person.name,
                    key=f"new_name_{person.person_id}"
                )
                if st.button("Save", key=f"save_name_{person.person_id}"):
                    # TODO: Implement rename API
                    st.session_state[f"show_rename_{person.person_id}"] = False
                    add_notification(f"Renamed to {new_name}", "success")
                    st.rerun()


def _render_pending_changes_tab(album_id: str, run_id: str) -> None:
    """Render tab showing pending batch changes."""
    st.subheader("Pending Changes")

    pending = st.session_state.get("fm_pending_changes", [])

    # Mode toggle
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio(
            "Mode",
            options=["batch", "live"],
            format_func=lambda x: "Batch (apply all at once)" if x == "batch" else "Live (apply immediately)",
            horizontal=True,
            key="fm_mode_toggle"
        )
        st.session_state.fm_mode = mode

    with col2:
        recluster = st.checkbox(
            "Re-cluster after apply",
            value=True,
            key="fm_recluster",
            help="Run identity refinement after applying changes"
        )

    st.divider()

    if not pending:
        st.info("No pending changes. Make changes in other tabs.")
        return

    st.write(f"**{len(pending)}** pending changes:")

    for i, change in enumerate(pending):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            action_desc = _describe_action(change)
            st.write(f"{i + 1}. {action_desc}")

        with col2:
            if st.button("Undo", key=f"undo_{i}"):
                st.session_state.fm_pending_changes.pop(i)
                add_notification("Change removed", "info")
                st.rerun()

        with col3:
            if st.button("Move up", key=f"up_{i}", disabled=(i == 0)):
                pending[i], pending[i - 1] = pending[i - 1], pending[i]
                st.rerun()

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Apply All Changes", type="primary", key="apply_all"):
            _apply_pending_changes(album_id, run_id, recluster)

    with col2:
        if st.button("Clear All", key="clear_all"):
            st.session_state.fm_pending_changes = []
            add_notification("All pending changes cleared", "info")
            st.rerun()

    with col3:
        st.write(f"Total: {len(pending)} changes")


def _describe_action(action: FaceAction) -> str:
    """Get human-readable description of an action."""
    face_short = action.face_key.split("/")[-1] if "/" in action.face_key else action.face_key

    if action.action == "assign":
        if action.new_person_name:
            return f"Assign {face_short} to new person '{action.new_person_name}'"
        else:
            return f"Assign {face_short} to {action.target_person_id}"
    elif action.action == "unassign":
        return f"Remove {face_short} from current person"
    elif action.action == "untag":
        return f"Untag {face_short} (don't care)"
    elif action.action == "not_a_face":
        return f"Mark {face_short} as not a face"
    else:
        return f"{action.action} on {face_short}"


def _add_pending_change(action: FaceAction) -> None:
    """Add a change to the pending list."""
    if "fm_pending_changes" not in st.session_state:
        st.session_state.fm_pending_changes = []

    # Remove any existing action for this face
    st.session_state.fm_pending_changes = [
        c for c in st.session_state.fm_pending_changes
        if c.face_key != action.face_key
    ]

    st.session_state.fm_pending_changes.append(action)

    # If in live mode, apply immediately
    if st.session_state.get("fm_mode") == "live":
        # Get current album and run
        state = get_session()
        if state.current_album:
            run_id = _get_active_run_id(state.current_album.album_id)
            if run_id:
                _apply_single_change(state.current_album.album_id, run_id, action)
                st.session_state.fm_pending_changes = []


def _apply_single_change(album_id: str, run_id: str, action: FaceAction) -> None:
    """Apply a single change immediately (live mode)."""
    client = get_client()
    result = client.apply_face_action(album_id, run_id, action)
    if result:
        add_notification(f"Applied: {_describe_action(action)}", "success")
    else:
        add_notification(f"Failed to apply action", "error")


def _apply_pending_changes(album_id: str, run_id: str, recluster: bool) -> None:
    """Apply all pending changes."""
    pending = st.session_state.get("fm_pending_changes", [])

    if not pending:
        add_notification("No changes to apply", "warning")
        return

    client = get_client()
    result = client.apply_batch_changes(album_id, run_id, pending, recluster=recluster)

    if result:
        msg = f"Applied {result.applied_count} changes"
        if result.auto_assigned_count > 0:
            msg += f", {result.auto_assigned_count} auto-assigned"
        if result.failed_count > 0:
            msg += f", {result.failed_count} failed"
            add_notification(msg, "warning")
        else:
            add_notification(msg, "success")

        # Clear pending changes
        st.session_state.fm_pending_changes = []
    else:
        add_notification("Failed to apply changes", "error")

    st.rerun()
