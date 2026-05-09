"""Image detail popup — tabbed view with bboxes, decisions, cluster peers, face info.

Triggered from any image in the app via image_detail_btn().
All data preloaded — zero API calls when popup opens.
"""
import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict

from app.streamlit.components.bbox_overlay import draw_face_bboxes


def image_detail_btn(image_info, key: str) -> None:
    """Render a button that opens the image detail popup when clicked."""
    if st.button("Detail", key=key, use_container_width=True):
        st.session_state["_image_popup_info"] = image_info
        st.rerun()


def maybe_show_image_popup() -> None:
    """Check session state and show popup if triggered. Call from main.py after page dispatch."""
    info = st.session_state.get("_image_popup_info")
    if info is not None:
        _show_image_dialog(info)


@st.dialog("Image Detail", width="large")
def _show_image_dialog(image_info) -> None:
    """Render the tabbed image detail dialog."""
    img_path = Path(image_info.path)

    # Header: filename + status
    hcol1, hcol2, hcol3 = st.columns([4, 2, 1])
    with hcol1:
        st.markdown(f"**`{img_path.name}`**")
    with hcol2:
        if image_info.is_selected:
            st.success("Selected")
        elif image_info.cluster_id is not None:
            st.error("Rejected")
        else:
            st.warning("Filtered")
    with hcol3:
        if st.button("X", key="popup_close_top"):
            st.session_state.pop("_image_popup_info", None)
            st.rerun()

    # Two columns: image left, tabs right
    col_img, col_tabs = st.columns([1.2, 1])

    with col_img:
        # Image with bounding boxes
        faces = _extract_faces_for_bbox(image_info)
        if faces:
            annotated = draw_face_bboxes(str(img_path), faces, max_size=500)
            if annotated:
                st.image(annotated, use_container_width=True)
            else:
                _show_thumbnail(img_path)
        else:
            _show_thumbnail(img_path)

        capts = []
        if image_info.face_count:
            capts.append(f"{image_info.face_count} face(s)")
        if image_info.cluster_id is not None:
            capts.append(f"Cluster {image_info.cluster_id}")
        st.caption(" | ".join(capts) if capts else img_path.name)

    with col_tabs:
        tab_q, tab_d, tab_s, tab_f, tab_det = st.tabs(
            ["Quality", "Decision", "Scene", "Faces", "Detection"]
        )

        with tab_q:
            _render_quality_tab(image_info)

        with tab_d:
            _render_decision_tab(image_info)

        with tab_s:
            _render_scene_tab(image_info)

        with tab_f:
            _render_faces_tab(image_info)

        with tab_det:
            _render_detection_tab(image_info)

    # Close button
    if st.button("Close", use_container_width=True, key="popup_close_bottom"):
        st.session_state.pop("_image_popup_info", None)
        st.rerun()


# ─── TAB: Quality ───

def _render_quality_tab(img) -> None:
    _score_row("IQA", img.iqa_score)
    _score_row("AVA", img.ava_score)
    _score_row("Sharpness", img.sharpness)
    st.divider()
    if img.composite_score is not None:
        st.metric("Composite Score", f"{img.composite_score:.2f}")
    else:
        st.write("Composite: --")


# ─── TAB: Decision ───

def _render_decision_tab(img) -> None:
    # Main decision
    if img.is_selected:
        st.success(f"**Selected** — Cluster {img.cluster_id}")
    elif img.cluster_id is not None:
        st.error(f"**Rejected** — Cluster {img.cluster_id}")
    else:
        st.warning("**Filtered out** before clustering")

    # Show StepDecision records for this image
    decisions = st.session_state.get("_popup_step_decisions") or []
    img_path_str = str(img.path).replace("\\", "/")
    my_decisions = [d for d in decisions if _paths_match(d.get("item_id", ""), img_path_str)]

    if my_decisions:
        st.caption("Pipeline decisions for this image:")
        for d in my_decisions:
            step = d.get("step", "?")
            decision = d.get("decision", "?")
            reason = d.get("reason", "")

            if "pass" in decision or "select" in decision or "detect" in decision:
                st.write(f":green[{step}] — {reason}")
            elif "reject" in decision or "fail" in decision or "noise" in decision:
                st.write(f":red[{step}] — {reason}")
            else:
                st.write(f":blue[{step}] — {reason}")

        # Show config from first decision
        cfg = my_decisions[0].get("config_used", {})
        if cfg:
            with st.expander("Config used"):
                st.json(cfg)
    else:
        st.caption("No pipeline decision records. Re-run pipeline to generate.")

        # Fallback: show basic reason
        if img.composite_score is not None and img.cluster_id is not None:
            st.write(f"Score: {img.composite_score:.2f} in cluster {img.cluster_id}")


# ─── TAB: Scene ───

def _render_scene_tab(img) -> None:
    if img.cluster_id is None:
        st.info("Not assigned to any scene cluster (filtered out)")
        return

    st.write(f"**Cluster {img.cluster_id}**")

    # Get peer images from session state
    all_images = st.session_state.get("_popup_all_images") or []
    peers = [i for i in all_images if i.cluster_id == img.cluster_id]
    selected_peers = [i for i in peers if i.is_selected]

    st.caption(f"{len(peers)} images in cluster | {len(selected_peers)} selected")

    if peers:
        # Show thumbnails in a grid
        n_show = min(len(peers), 10)
        cols = st.columns(min(n_show, 5))
        for i, peer in enumerate(peers[:n_show]):
            with cols[i % 5]:
                from app.streamlit.pages.explore import _load_thumb
                thumb = _load_thumb(peer.path, size=60)
                if thumb:
                    st.image(thumb, width=60)
                is_current = peer.path == img.path
                is_sel = peer.is_selected
                if is_current:
                    st.caption(":red[Current]")
                elif is_sel:
                    st.caption(":green[Selected]")
                else:
                    st.caption(Path(peer.path).name[:8])

        if len(peers) > n_show:
            st.caption(f"... and {len(peers) - n_show} more")


# ─── TAB: Faces ───

def _render_faces_tab(img) -> None:
    if not img.face_count or img.face_count == 0:
        st.info("No faces detected in this image")
        return

    filter_scores = img.filter_scores or []
    pose_scores = img.face_pose_scores or []
    eyes_scores = img.face_eyes_scores or []
    smile_scores = img.face_smile_scores or []

    n_faces = max(img.face_count, len(filter_scores), len(pose_scores))

    for i in range(n_faces):
        fs = filter_scores[i] if i < len(filter_scores) else {}
        conf = fs.get("confidence")
        passed = fs.get("filter_passed", True)

        st.markdown(f"**Face #{i}**" + (" :red[(filtered)]" if not passed else ""))

        # Scores as compact chips
        parts = []
        if conf is not None:
            parts.append(f"det: {conf:.2f}")
        if i < len(pose_scores):
            v = pose_scores[i]
            color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
            parts.append(f"pose: :{color}[{v:.2f}]")
        if i < len(eyes_scores):
            v = eyes_scores[i]
            color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
            parts.append(f"eyes: :{color}[{v:.2f}]")
        if i < len(smile_scores):
            v = smile_scores[i]
            color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
            parts.append(f"expr: :{color}[{v:.2f}]")

        st.write(" | ".join(parts) if parts else "No scores available")

        # Provenance: cache key + bbox coords
        path_norm = str(img.path).replace("\\", "/")
        cache_key = f"{path_norm}:face_{i}"
        bbox = fs.get("bbox", {})
        if bbox:
            coords = f"x={bbox.get('x', bbox.get('x_px', '?'))}, y={bbox.get('y', bbox.get('y_px', '?'))}"
        else:
            coords = "no bbox"
        st.caption(f"Cache: `{Path(cache_key).name}` | {coords}")

        if i < n_faces - 1:
            st.divider()


# ─── TAB: Detection ───

def _render_detection_tab(img) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if img.person_detected is not None:
            if img.person_detected:
                st.metric("Person Detected", "Yes")
            else:
                st.metric("Person Detected", "No")
        else:
            st.metric("Person Detected", "--")

        if img.person_confidence is not None:
            st.metric("Confidence", f"{img.person_confidence:.2f}")

    with col2:
        if img.body_facing_score is not None:
            st.metric("Body Facing", f"{img.body_facing_score:.2f}")

    # Show face bbox coordinates
    filter_scores = img.filter_scores or []
    if filter_scores:
        st.divider()
        st.caption("Face bounding boxes:")
        for fs in filter_scores:
            bbox = fs.get("bbox", {})
            idx = fs.get("face_index", 0)
            if bbox:
                coords = []
                if bbox.get("x") is not None:
                    coords = [f"x={bbox['x']:.2f}", f"y={bbox['y']:.2f}",
                              f"w={bbox['w']:.2f}", f"h={bbox['h']:.2f}"]
                elif bbox.get("x_px") is not None:
                    coords = [f"x={bbox['x_px']}px", f"y={bbox['y_px']}px",
                              f"w={bbox['w_px']}px", f"h={bbox['h_px']}px"]
                conf = fs.get("confidence")
                st.write(f"Face {idx}: {', '.join(coords)}" + (f" | conf={conf:.2f}" if conf else ""))


# ─── Helpers ───

def _score_row(label: str, value: Optional[float]) -> None:
    if value is None:
        st.write(f"{label}: --")
        return
    color = "green" if value >= 0.6 else ("orange" if value >= 0.3 else "red")
    st.write(f"{label}: :{color}[**{value:.2f}**]")


def _extract_faces_for_bbox(image_info) -> list:
    """Extract face bbox data from image_info for drawing."""
    faces = []
    filter_scores = image_info.filter_scores or []
    for fs in filter_scores:
        bbox = fs.get("bbox")
        if not bbox:
            continue
        has_px = bbox.get("w_px", 0) > 0
        has_norm = bbox.get("w", 0) > 0
        if has_px or has_norm:
            faces.append({
                "bbox": bbox,
                "label": f"face_{fs.get('face_index', 0)}",
                "confidence": fs.get("confidence"),
            })
    return faces


def _show_thumbnail(img_path: Path) -> None:
    try:
        from PIL import Image, ImageOps
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((500, 500))
            st.image(img, use_container_width=True)
    except Exception:
        st.warning(f"Could not load {img_path.name}")


def _paths_match(path_a: str, path_b: str) -> bool:
    """Check if two paths refer to the same file (handle slash differences)."""
    return Path(path_a).name == Path(path_b).name
