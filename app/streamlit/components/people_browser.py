"""People browser component - Google Photos style."""

import logging
import streamlit as st
from pathlib import Path
from typing import List, Optional, Callable, Set
from PIL import Image, ImageOps

from app.streamlit.models import Person, ImageInfo
from app.streamlit.api_client import get_client
from app.streamlit.config import get_config
from app.streamlit.session import add_notification

logger = logging.getLogger(__name__)


def render_people_grid(
    people: List[Person],
    columns: Optional[int] = None,
    on_person_click: Optional[Callable[[Person], None]] = None,
    album_id: Optional[str] = None,
    enable_selection: bool = False,
) -> Optional[Person]:
    """Render a grid of people with face thumbnails."""
    if not people:
        st.info("No people detected in this album")
        return None

    config = get_config()
    num_cols = columns or config.people_images_per_row
    sorted_people = sorted(people, key=lambda p: p.face_count, reverse=True)

    selected_person = None
    cols = st.columns(num_cols)

    for i, person in enumerate(sorted_people):
        with cols[i % num_cols]:
            if enable_selection:
                # Show checkbox for merge selection
                is_selected = st.checkbox(
                    "Select",
                    key=f"select_{person.person_id}",
                    label_visibility="collapsed"
                )
                if is_selected:
                    _add_to_merge_selection(person.person_id)
                else:
                    _remove_from_merge_selection(person.person_id)

            if render_person_card(person, album_id):
                selected_person = person
                if on_person_click:
                    on_person_click(person)

    return selected_person


def _add_to_merge_selection(person_id: str) -> None:
    """Add person to merge selection."""
    if "merge_selection" not in st.session_state:
        st.session_state.merge_selection = set()
    st.session_state.merge_selection.add(person_id)


def _remove_from_merge_selection(person_id: str) -> None:
    """Remove person from merge selection."""
    if "merge_selection" in st.session_state:
        st.session_state.merge_selection.discard(person_id)


def render_person_card(person: Person, album_id: Optional[str] = None) -> bool:
    """Render a single person card. Returns True if View clicked."""
    _render_person_thumbnail(person)

    display_name = person.name or f"Person {person.person_index + 1}"
    rename_key = f"renaming_{person.person_id}"

    # Inline rename mode
    if st.session_state.get(rename_key):
        new_name = st.text_input("Name", value=person.name or "", key=f"input_{person.person_id}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", key=f"save_{person.person_id}"):
                if new_name and album_id:
                    try:
                        get_client().rename_person(album_id, person.person_id, new_name)
                        add_notification(f"Renamed to {new_name}", "success")
                    except Exception as e:
                        add_notification(f"Rename failed: {e}", "error")
                elif not album_id:
                    add_notification("Cannot rename: no album selected", "error")
                st.session_state[rename_key] = False
                st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel_{person.person_id}"):
                st.session_state[rename_key] = False
                st.rerun()
        return False

    # Normal display mode
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{display_name}**")
    with col2:
        if st.button("✏️", key=f"edit_{person.person_id}", help="Rename"):
            st.session_state[rename_key] = True
            st.rerun()

    st.caption(f"{person.face_count} photos")
    return st.button("View", key=f"view_{person.person_id}")


def _load_face_thumbnail(person: Person) -> Optional[Image.Image]:
    """Load and return person's face thumbnail as a PIL Image, or None on failure."""
    if not person.representative_face:
        return None
    face_path = Path(person.representative_face)
    if not face_path.exists():
        return None
    try:
        img = ImageOps.exif_transpose(Image.open(face_path))
        if person.thumbnail_bbox and len(person.thumbnail_bbox) == 4:
            img = _crop_face(img, person.thumbnail_bbox)
        return img
    except Exception:
        # SIGHTING-104: this used to swallow the error silently and fall back to
        # a gray placeholder, hiding the pixel-vs-normalized bbox bug for weeks.
        # Log it so the next failure is visible instead of mysterious.
        logger.warning(
            "Failed to load face thumbnail for person %s (path=%s, bbox=%s)",
            getattr(person, "person_id", "?"), face_path, person.thumbnail_bbox,
            exc_info=True,
        )
        return None


def _render_person_thumbnail(person: Person) -> None:
    """Render person's face thumbnail."""
    img = _load_face_thumbnail(person)
    if img:
        st.image(img, use_container_width=True)
    else:
        _render_placeholder()


def _crop_face(img: Image.Image, bbox: list) -> Image.Image:
    """Crop image to face region with padding.

    ``bbox`` is expected normalized ``[x, y, w, h]`` in [0, 1]. SIGHTING-104:
    legacy/spec-079 rows persisted pixel-scale bboxes, which blew the crop
    off-canvas (-> gray placeholder). Guard by detecting pixel-scale values and
    normalizing them here so existing DB rows render without a re-run.
    """
    img_w, img_h = img.size
    bx, by, bw, bh = bbox
    if max(bx, by, bw, bh) > 1.5:  # pixel-scale -> normalize
        bx, by, bw, bh = bx / img_w, by / img_h, bw / img_w, bh / img_h
    x, y, w, h = [v * d for v, d in zip([bx, by, bw, bh], [img_w, img_h, img_w, img_h])]

    pad = 0.3
    left = max(0, int(x - w * pad))
    top = max(0, int(y - h * pad))
    right = min(img_w, int(x + w * (1 + pad)))
    bottom = min(img_h, int(y + h * (1 + pad)))

    return img.crop((left, top, right, bottom))


def _render_placeholder() -> None:
    """Render placeholder avatar."""
    st.markdown(
        '<div style="width:100%;aspect-ratio:1;background:#e0e0e0;border-radius:50%;'
        'display:flex;align-items:center;justify-content:center;font-size:2rem;">👤</div>',
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
    display_name = person.name or f"Person {person.person_index + 1}"

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if on_back and st.button("← Back"):
            on_back()
    with col2:
        st.header(display_name)
    with col3:
        if st.button("✏️ Rename", key="detail_rename"):
            st.session_state.renaming_person = person.person_id
            st.rerun()

    if st.session_state.get("renaming_person") == person.person_id:
        _render_rename_dialog(person, album_id, on_rename)

    st.caption(f"{person.face_count} appearances in {person.image_count} photos")

    # Show representative image with face bounding box
    if person.representative_face and person.thumbnail_bbox and len(person.thumbnail_bbox) == 4:
        from app.streamlit.components.bbox_overlay import draw_face_bboxes
        bx, by, bw, bh = person.thumbnail_bbox
        # SIGHTING-104: pixel-scale rows must use the pixel keys (x_px/...) that
        # draw_face_bboxes understands; normalized rows use x/y/w/h.
        if max(person.thumbnail_bbox) > 1.5:
            bbox_dict = {"x_px": bx, "y_px": by, "w_px": bw, "h_px": bh}
        else:
            bbox_dict = {"x": bx, "y": by, "w": bw, "h": bh}
        faces = [{"bbox": bbox_dict, "label": display_name, "confidence": None}]
        annotated = draw_face_bboxes(person.representative_face, faces, highlight_index=0, max_size=300)
        if annotated:
            st.image(annotated, caption="Face highlighted in source image", width=300)

    st.divider()

    _render_person_images(person, album_id)


def _render_person_images(person: Person, album_id: str) -> None:
    """Render person's images with filters."""
    images = get_client().get_images_by_person(album_id, person.person_id)
    if not images:
        st.info("No images found for this person")
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_mode = st.selectbox("Filter", ["All", "Solo", "With others", "Selected"], key="filter")
    with col2:
        sort_by = st.selectbox("Sort", ["Score", "Face count", "Filename"], key="sort")
    with col3:
        columns = st.slider("Cols", 2, 6, 4, key="cols")

    # Apply filters
    if filter_mode == "Solo":
        images = [i for i in images if i.face_count == 1]
    elif filter_mode == "With others":
        images = [i for i in images if i.face_count > 1]
    elif filter_mode == "Selected":
        images = [i for i in images if i.is_selected]

    # Apply sort
    if sort_by == "Score":
        images = sorted(images, key=lambda x: x.composite_score or 0, reverse=True)
    elif sort_by == "Face count":
        images = sorted(images, key=lambda x: x.face_count, reverse=True)
    elif sort_by == "Filename":
        images = sorted(images, key=lambda x: x.filename)

    st.write(f"**{len(images)}** photos")

    from app.streamlit.components.gallery import render_image_gallery
    render_image_gallery(images, show_scores=True, show_selection=True, columns=columns)


def _render_rename_dialog(person: Person, album_id: str, on_rename: Optional[Callable[[str], None]] = None) -> None:
    """Render rename dialog."""
    with st.form("rename_form"):
        new_name = st.text_input("New Name", value=person.name or "")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Save", type="primary") and new_name:
                get_client().rename_person(album_id, person.person_id, new_name)
                st.session_state.renaming_person = None
                add_notification(f"Renamed to {new_name}", "success")
                if on_rename:
                    on_rename(new_name)
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.renaming_person = None
                st.rerun()


def render_people_summary_row(people: List[Person]) -> None:
    """Render a compact summary row of detected people."""
    if not people:
        return

    st.write("**People detected:**")
    cols = st.columns(min(len(people), 8) + 1)

    for i, person in enumerate(people[:8]):
        with cols[i]:
            img = _load_face_thumbnail(person)
            if img:
                st.image(img, width=50)
            else:
                st.write("👤")
            st.caption((person.name or f"#{i+1}")[:10])

    if len(people) > 8:
        with cols[8]:
            st.write(f"+{len(people) - 8}")


def render_merge_dialog(
    people: List[Person],
    album_id: str,
    on_complete: Optional[Callable[[], None]] = None,
) -> None:
    """Render dialog to merge selected people."""
    selected_ids: Set[str] = st.session_state.get("merge_selection", set())

    if len(selected_ids) < 2:
        st.warning("Select at least 2 people to merge")
        return

    selected_people = [p for p in people if p.person_id in selected_ids]

    st.subheader("Merge People")
    st.write(f"Merging **{len(selected_people)}** people into one:")

    cols = st.columns(min(len(selected_people), 4))
    for i, person in enumerate(selected_people[:4]):
        with cols[i]:
            _render_person_thumbnail(person)
            st.caption(person.name or f"Person {person.person_index + 1}")

    if len(selected_people) > 4:
        st.caption(f"... and {len(selected_people) - 4} more")

    st.divider()

    with st.form("merge_form"):
        new_name = st.text_input(
            "Name for merged person",
            value=selected_people[0].name or "",
            placeholder="Enter a name..."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Merge", type="primary"):
                try:
                    person_ids = [p.person_id for p in selected_people]
                    merged = get_client().merge_people(album_id, person_ids)
                    if new_name:
                        get_client().rename_person(album_id, merged.person_id, new_name)
                    add_notification(f"Merged {len(selected_people)} people", "success")
                    st.session_state.merge_selection = set()
                    if on_complete:
                        on_complete()
                    st.rerun()
                except Exception as e:
                    add_notification(f"Merge failed: {e}", "error")

        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.merge_selection = set()
                if on_complete:
                    on_complete()
                st.rerun()
