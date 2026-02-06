"""People browser component - Google Photos style."""

import streamlit as st
from pathlib import Path
from typing import List, Optional, Callable, Set
from PIL import Image, ImageOps

from app.streamlit.models import Person, ImageInfo
from app.streamlit.api_client import get_client, ApiError
from app.streamlit.config import get_config
from app.streamlit.session import add_notification


def render_people_grid(
    people: List[Person],
    columns: Optional[int] = None,
    on_person_click: Optional[Callable[[Person], None]] = None,
    enable_selection: bool = False,
    album_id: Optional[str] = None,
) -> Optional[Person]:
    """Render a grid of people with face thumbnails."""
    if not people:
        st.info("No people detected in this album")
        return None

    config = get_config()
    num_cols = columns or config.people_images_per_row
    sorted_people = sorted(people, key=lambda p: p.face_count, reverse=True)

    if enable_selection:
        selected_ids = _render_selection_toolbar(people, album_id)
    else:
        selected_ids = set()

    selected_person = None
    cols = st.columns(num_cols)

    for i, person in enumerate(sorted_people):
        with cols[i % num_cols]:
            if enable_selection:
                is_selected = person.person_id in selected_ids
                if render_person_card_selectable(person, is_selected):
                    if person.person_id in selected_ids:
                        selected_ids.discard(person.person_id)
                    else:
                        selected_ids.add(person.person_id)
                    st.session_state.selected_person_ids = selected_ids
                    st.rerun()
            else:
                if render_person_card(person, album_id=album_id):
                    selected_person = person
                    if on_person_click:
                        on_person_click(person)

    return selected_person


def _render_selection_toolbar(people: List[Person], album_id: Optional[str]) -> Set[str]:
    """Render toolbar for multi-select mode."""
    selected_ids = st.session_state.get("selected_person_ids", set())

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("Select All", key="select_all_people"):
            st.session_state.selected_person_ids = {p.person_id for p in people}
            st.rerun()

    with col2:
        if st.button("Clear", key="clear_selection"):
            st.session_state.selected_person_ids = set()
            st.rerun()

    with col3:
        st.write(f"**{len(selected_ids)}** selected")

    with col4:
        if len(selected_ids) >= 2 and album_id:
            if st.button("Merge Selected", type="primary", key="merge_people_btn"):
                client = get_client()
                client.merge_people(album_id, list(selected_ids))
                add_notification(f"Merged {len(selected_ids)} people", "success")
                st.session_state.selected_person_ids = set()
                st.rerun()

    st.divider()
    return selected_ids


def render_person_card(person: Person, album_id: Optional[str] = None) -> bool:
    """Render a single person card. Returns True if clicked."""
    with st.container():
        _render_person_thumbnail(person)
        display_name = person.name or f"Person {person.person_id[:6]}"

        # Inline rename functionality
        rename_key = f"renaming_inline_{person.person_id}"
        if st.session_state.get(rename_key):
            new_name = st.text_input(
                "Name",
                value=person.name or "",
                key=f"name_input_{person.person_id}",
                placeholder="Enter name"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", key=f"save_name_{person.person_id}", use_container_width=True):
                    if new_name and album_id:
                        client = get_client()
                        client.rename_person(album_id, person.person_id, new_name)
                        add_notification(f"Renamed to {new_name}", "success")
                    st.session_state[rename_key] = False
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_name_{person.person_id}", use_container_width=True):
                    st.session_state[rename_key] = False
                    st.rerun()
            return False
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{display_name}**")
            with col2:
                if st.button("✏️", key=f"edit_name_{person.person_id}", help="Rename"):
                    st.session_state[rename_key] = True
                    st.rerun()
            st.caption(f"{person.face_count} photos")
            return st.button("View", key=f"person_{person.person_id}", use_container_width=True)


def render_person_card_selectable(person: Person, is_selected: bool) -> bool:
    """Render a person card with selection checkbox. Returns True if selection changed."""
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            checkbox_clicked = st.checkbox("", value=is_selected, key=f"select_person_{person.person_id}", label_visibility="collapsed")

        _render_person_thumbnail(person)
        display_name = person.name or f"Person {person.person_id[:6]}"
        st.write(f"**{display_name}**" if is_selected else display_name)
        st.caption(f"{person.face_count} photos")

        return checkbox_clicked != is_selected


def _render_person_thumbnail(person: Person) -> None:
    """Render person's representative face thumbnail."""
    if not person.representative_face:
        _render_placeholder_avatar()
        return

    face_path = Path(person.representative_face)
    if not face_path.exists():
        _render_placeholder_avatar()
        return

    try:
        img = Image.open(face_path)
        img = ImageOps.exif_transpose(img)
        img = _crop_face_region(img, person.thumbnail_bbox)
        st.image(img, use_container_width=True)
    except Exception:
        _render_placeholder_avatar()


def _crop_face_region(img: Image.Image, bbox: list) -> Image.Image:
    """Crop image to face region with padding. Returns original if no bbox."""
    if not bbox or len(bbox) != 4:
        return img

    img_w, img_h = img.size
    x, y, w, h = [v * dim for v, dim in zip(bbox, [img_w, img_h, img_w, img_h])]

    # 30% padding
    pad_x, pad_y = w * 0.3, h * 0.3
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(img_w, int(x + w + pad_x))
    bottom = min(img_h, int(y + h + pad_y))

    return img.crop((left, top, right, bottom))


def _render_placeholder_avatar() -> None:
    """Render a placeholder avatar."""
    st.markdown(
        '<div style="width:100%;aspect-ratio:1;background:#e0e0e0;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;">👤</div>',
        unsafe_allow_html=True,
    )


def render_person_detail(
    person: Person,
    album_id: str,
    on_back: Optional[Callable[[], None]] = None,
    on_rename: Optional[Callable[[str], None]] = None,
    all_people: Optional[List[Person]] = None,
) -> None:
    """Render detailed view of a person with all their images."""
    display_name = person.name or f"Person {person.person_id[:6]}"

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if on_back and st.button("← Back"):
            on_back()

    with col2:
        st.header(display_name)

    with col3:
        if st.button("✏️ Rename", key="rename_btn"):
            st.session_state.renaming_person = person.person_id
            st.rerun()

    if st.session_state.get("renaming_person") == person.person_id:
        _render_rename_dialog(person, album_id, on_rename)

    st.caption(f"{person.face_count} appearances in {person.image_count} photos")
    st.divider()

    _render_person_images_with_filters(person, album_id, display_name)


def _render_person_images_with_filters(person: Person, album_id: str, display_name: str) -> None:
    """Render person's images with filter options."""
    client = get_client()
    images = client.get_images_by_person(album_id, person.person_id)

    if not images:
        st.info("No images found for this person")
        return

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        filter_mode = st.selectbox("Filter", ["All photos", "Solo (alone)", "With others", "Selected only"], key="person_filter_mode")

    with col2:
        sort_by = st.selectbox("Sort by", ["Score (best first)", "Face count", "Filename"], key="person_sort_by")

    with col3:
        columns = st.slider("Columns", 2, 6, 4, key="person_gallery_cols")

    with col4:
        show_scores = st.checkbox("Scores", value=True, key="person_show_scores")

    filtered_images = _apply_person_image_filters(images, filter_mode)
    filtered_images = _apply_person_image_sort(filtered_images, sort_by)

    st.write(f"**{len(filtered_images)}** of {len(images)} photos")

    if not filtered_images:
        st.info(f"No photos match the '{filter_mode}' filter")
        return

    from app.streamlit.components.gallery import render_image_gallery
    render_image_gallery(filtered_images, show_scores=show_scores, show_selection=True, columns=columns)


def _apply_person_image_filters(images: List[ImageInfo], filter_mode: str) -> List[ImageInfo]:
    """Apply filter to person's images."""
    if filter_mode == "Solo (alone)":
        return [img for img in images if img.face_count == 1]
    elif filter_mode == "With others":
        return [img for img in images if img.face_count > 1]
    elif filter_mode == "Selected only":
        return [img for img in images if img.is_selected]
    return images


def _apply_person_image_sort(images: List[ImageInfo], sort_by: str) -> List[ImageInfo]:
    """Apply sorting to images."""
    if sort_by == "Score (best first)":
        return sorted(images, key=lambda x: x.composite_score or x.ava_score or x.iqa_score or 0, reverse=True)
    elif sort_by == "Face count":
        return sorted(images, key=lambda x: x.face_count, reverse=True)
    elif sort_by == "Filename":
        return sorted(images, key=lambda x: x.filename)
    return images


def _render_rename_dialog(person: Person, album_id: str, on_rename: Optional[Callable[[str], None]] = None) -> None:
    """Render rename dialog for a person."""
    with st.form("rename_person_form"):
        new_name = st.text_input("New Name", value=person.name or "", placeholder="Enter name")

        col1, col2 = st.columns(2)

        with col1:
            if st.form_submit_button("Save", type="primary") and new_name:
                client = get_client()
                client.rename_person(album_id, person.person_id, new_name)
                st.session_state.renaming_person = None
                add_notification(f"Renamed to {new_name}", "success")
                if on_rename:
                    on_rename(new_name)
                st.rerun()

        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.renaming_person = None
                st.rerun()


def render_split_dialog(person: Person, album_id: str, on_complete: Optional[Callable[[], None]] = None) -> None:
    """Render dialog for splitting faces from a person."""
    st.subheader(f"Split faces from {person.name or 'Person'}")
    st.info("Select faces that belong to a **different person** and split them into a new identity.")

    with st.form("split_faces_form"):
        face_indices_str = st.text_input("Face indices to split (comma-separated)", placeholder="0, 1, 2")
        new_person_name = st.text_input("Name for new person (optional)", placeholder="Leave blank to auto-generate")

        col1, col2 = st.columns(2)

        with col1:
            if st.form_submit_button("Split", type="primary") and face_indices_str:
                indices = [int(x.strip()) for x in face_indices_str.split(",") if x.strip().isdigit()]
                if indices:
                    client = get_client()
                    client.split_person(album_id, person.person_id, indices)
                    add_notification(f"Split {len(indices)} faces into new person", "success")
                    st.session_state.show_split_dialog = False
                    if on_complete:
                        on_complete()
                    st.rerun()

        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.show_split_dialog = False
                st.rerun()


def render_merge_dialog(people: List[Person], album_id: str, on_complete: Optional[Callable[[], None]] = None) -> None:
    """Render dialog for merging multiple people."""
    st.subheader("Merge People")
    st.info("Select two or more people to merge into a single identity.")

    person_options = {f"{p.name or f'Person {p.person_id[:6]}'} ({p.face_count} photos)": p.person_id for p in people}
    selected_labels = st.multiselect("Select people to merge", options=list(person_options.keys()), key="merge_people_select")
    selected_ids = [person_options[label] for label in selected_labels]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Merge", type="primary", disabled=len(selected_ids) < 2):
            client = get_client()
            client.merge_people(album_id, selected_ids)
            add_notification(f"Merged {len(selected_ids)} people", "success")
            if on_complete:
                on_complete()
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.rerun()

    if len(selected_ids) < 2:
        st.caption("Select at least 2 people to merge")


def render_people_filter(people: List[Person], selected_ids: List[str], on_filter_change: Optional[Callable[[List[str]], None]] = None) -> List[str]:
    """Render people filter chips for filtering images."""
    st.write("**Filter by person:**")

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("All", key="people_filter_all"):
            selected_ids = [p.person_id for p in people]
            if on_filter_change:
                on_filter_change(selected_ids)
            st.rerun()

    with col2:
        if st.button("None", key="people_filter_none"):
            selected_ids = []
            if on_filter_change:
                on_filter_change(selected_ids)
            st.rerun()

    cols = st.columns(min(len(people), 6))

    for i, person in enumerate(people[:6]):
        display_name = person.name or f"Person {i+1}"
        is_selected = person.person_id in selected_ids

        with cols[i % 6]:
            if st.checkbox(display_name, value=is_selected, key=f"people_filter_{person.person_id}"):
                if person.person_id not in selected_ids:
                    selected_ids.append(person.person_id)
            elif person.person_id in selected_ids:
                selected_ids.remove(person.person_id)

    if on_filter_change:
        on_filter_change(selected_ids)

    return selected_ids


def render_people_summary_row(people: List[Person]) -> None:
    """Render a compact summary row of detected people."""
    if not people:
        return

    st.write("**People detected:**")

    cols = st.columns(min(len(people), 8) + 1)

    for i, person in enumerate(people[:8]):
        with cols[i]:
            if person.representative_face:
                face_path = Path(person.representative_face)
                if face_path.exists():
                    st.image(str(face_path), width=50)
                else:
                    st.write("👤")
            else:
                st.write("👤")
            st.caption((person.name or f"#{i+1}")[:10])

    if len(people) > 8:
        with cols[8]:
            st.write(f"+{len(people) - 8}")
