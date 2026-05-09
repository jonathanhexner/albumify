"""Explore page - Per-step pipeline observability.

Reads StepDecision records from the API. Zero pipeline logic in this file.
Shows image thumbnails for visual inspection.
"""
import io
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

from app.streamlit.session import get_session
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.api_client import get_client
from app.streamlit.components.image_popup import image_detail_btn

logger = logging.getLogger(__name__)

PAGE_SIZE = 20


@st.cache_data(show_spinner=False)
def _load_thumb(path: str, size: int = 60) -> Optional[str]:
    """Load a square thumbnail as base64."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            sq = min(img.size)
            left, top = (img.width - sq) // 2, (img.height - sq) // 2
            img = img.crop((left, top, left + sq, top + sq))
            img = img.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def render_explore_page() -> None:
    st.header("Explore")
    state = get_session()
    if not state.api_connected:
        st.warning("Connect to API to explore pipeline results.")
        return

    album = render_album_selector()
    if not album:
        st.info("Select an album to explore pipeline results.")
        return

    client = get_client()
    results = client.list_results(album.album_id)
    if not results:
        st.info("No pipeline results yet. Go to Configure & Run to run the pipeline.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))
    decisions = latest.get("step_decisions") or []
    images = client.get_images(job_id)
    st.session_state["_popup_all_images"] = images
    # Build path->ImageInfo lookup for thumbnails
    img_lookup = {img.path: img for img in images} if images else {}

    # Store decisions + images in session state for popup access
    st.session_state["_popup_step_decisions"] = decisions

    st.caption(f"Run `{job_id[:8]}...` | {latest.get('total_images', 0)} images | {latest.get('total_duration_ms', 0)/1000:.1f}s")
    if not decisions:
        st.warning("No step decisions for this run. Re-run pipeline to generate decision data.")
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Image Quality", "Person Detection", "Face Detection",
        "Scene Clustering", "Face Clustering", "Selection", "Face Distance",
    ])

    with tab1:
        _render_step_tab(decisions, "filter_quality", "Image Quality Filtering",
                         "Which images passed/failed quality thresholds", img_lookup, images)
    with tab2:
        _render_step_tab(decisions, "detect_persons", "Person Detection",
                         "YOLOv8-Pose detection results", img_lookup, images)
    with tab3:
        _render_face_detection_tab(decisions, img_lookup, images)
    with tab4:
        _render_scene_clustering_tab(decisions, job_id, img_lookup, images)
    with tab5:
        _render_face_clustering_tab(decisions, job_id, latest, album)
    with tab6:
        _render_step_tab(decisions, "select_best", "Selection Decisions",
                         "Why each image was selected or rejected", img_lookup, images)
    with tab7:
        _render_face_distance_tab(job_id, images)


def _render_step_tab(
    all_decisions: List[Dict], step_name: str, title: str,
    subtitle: str, img_lookup: Dict, images: list,
) -> None:
    st.subheader(title)
    st.caption(subtitle)

    step_decisions = [d for d in all_decisions if d.get("step") == step_name]

    if not step_decisions:
        if images:
            _render_fallback_with_thumbs(images, step_name, img_lookup)
        else:
            st.info(f"No data for '{step_name}'.")
        return

    # Summary metrics
    counts = {}
    for d in step_decisions:
        dec = d.get("decision", "?")
        counts[dec] = counts.get(dec, 0) + 1

    cols = st.columns(min(len(counts) + 1, 6))
    cols[0].metric("Total", len(step_decisions))
    for i, (dec, count) in enumerate(sorted(counts.items()), 1):
        if i < len(cols):
            cols[i].metric(dec.replace("_", " ").title(), count)

    # Config used
    cfg = step_decisions[0].get("config_used", {})
    if cfg:
        with st.expander("Config used"):
            st.json(cfg)

    # Visual decision list with thumbnails
    _render_decisions_with_thumbs(step_decisions, img_lookup, step_name)


def _render_decisions_with_thumbs(decisions: List[Dict], img_lookup: Dict, step_name: str) -> None:
    """Render decisions as visual rows with image thumbnails."""
    page = st.session_state.get(f"explore_page_{step_name}", 0)
    total = len(decisions)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    showing = decisions[start:end]

    st.caption(f"Showing {start+1}-{end} of {total}")

    # Header
    hcols = st.columns([1, 3, 2, 3, 4])
    hcols[0].markdown("**Img**")
    hcols[1].markdown("**File**")
    hcols[2].markdown("**Decision**")
    hcols[3].markdown("**Key Metrics**")
    hcols[4].markdown("**Reason**")

    for d in showing:
        item_id = d.get("item_id", "")
        # Resolve full path from img_lookup
        img_info = _find_image(item_id, img_lookup)
        full_path = img_info.path if img_info else item_id
        filename = Path(item_id).name

        cols = st.columns([1, 3, 2, 3, 4])

        # Thumbnail
        with cols[0]:
            thumb = _load_thumb(full_path) if img_info else None
            if thumb:
                st.image(thumb, width=50)
            else:
                st.write("--")

        # Filename
        with cols[1]:
            st.write(filename)
            if img_info:
                image_detail_btn(img_info, key=f"det_{step_name}_{filename}")

        # Decision badge
        with cols[2]:
            dec = d.get("decision", "")
            if "pass" in dec or "select" in dec or "detect" in dec:
                st.success(dec)
            elif "reject" in dec or "fail" in dec or "noise" in dec or "not_" in dec:
                st.error(dec)
            else:
                st.info(dec)

        # Key metrics
        with cols[3]:
            metrics = d.get("metrics", {})
            parts = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.2f}")
                elif isinstance(v, int):
                    parts.append(f"{k}: {v}")
            st.caption(" | ".join(parts[:4]) if parts else "--")

        # Reason
        with cols[4]:
            st.write(d.get("reason", "--"))

    # Pagination
    if total > PAGE_SIZE:
        pcols = st.columns(3)
        with pcols[0]:
            if page > 0 and st.button("Previous", key=f"prev_{step_name}"):
                st.session_state[f"explore_page_{step_name}"] = page - 1
                st.rerun()
        with pcols[1]:
            st.caption(f"Page {page+1}/{(total-1)//PAGE_SIZE+1}")
        with pcols[2]:
            if end < total and st.button("Next", key=f"next_{step_name}"):
                st.session_state[f"explore_page_{step_name}"] = page + 1
                st.rerun()


def _render_fallback_with_thumbs(images: list, step_name: str, img_lookup: Dict) -> None:
    """Fallback when no step_decisions — show images with basic data + thumbnails."""
    st.info("No structured decisions. Showing raw image data. Re-run pipeline for decision records.")

    page = st.session_state.get(f"explore_fb_page_{step_name}", 0)
    total = len(images)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    showing = images[start:end]

    st.caption(f"Showing {start+1}-{end} of {total}")

    for img in showing:
        cols = st.columns([1, 3, 2, 3, 4])
        with cols[0]:
            thumb = _load_thumb(img.path)
            if thumb:
                st.image(thumb, width=50)
            else:
                st.write("--")
        with cols[1]:
            st.write(Path(img.path).name)
            image_detail_btn(img, key=f"fb_{step_name}_{Path(img.path).name}")
        with cols[2]:
            if img.is_selected:
                st.success("Selected")
            else:
                st.error("Not selected")
        with cols[3]:
            parts = []
            if step_name == "filter_quality":
                if img.iqa_score is not None: parts.append(f"IQA: {img.iqa_score:.2f}")
                if img.sharpness is not None: parts.append(f"Sharp: {img.sharpness:.2f}")
            elif step_name == "detect_persons":
                if img.person_detected is not None:
                    parts.append(f"Person: {'Yes' if img.person_detected else 'No'}")
                if img.person_confidence is not None:
                    parts.append(f"Conf: {img.person_confidence:.2f}")
            elif step_name == "select_best":
                if img.composite_score is not None: parts.append(f"Score: {img.composite_score:.2f}")
                if img.cluster_id is not None: parts.append(f"Cluster: {img.cluster_id}")
            else:
                parts.append(f"Faces: {img.face_count}")
            st.caption(" | ".join(parts) if parts else "--")
        with cols[4]:
            st.write("--")

    # Pagination
    if total > PAGE_SIZE:
        pcols = st.columns(3)
        with pcols[0]:
            if page > 0 and st.button("Prev", key=f"fb_prev_{step_name}"):
                st.session_state[f"explore_fb_page_{step_name}"] = page - 1
                st.rerun()
        with pcols[1]:
            st.caption(f"Page {page+1}/{(total-1)//PAGE_SIZE+1}")
        with pcols[2]:
            if end < total and st.button("Next", key=f"fb_next_{step_name}"):
                st.session_state[f"explore_fb_page_{step_name}"] = page + 1
                st.rerun()


def _render_face_detection_tab(all_decisions, img_lookup, images):
    """Face detection tab — shows images with bounding boxes around detected faces."""
    from app.streamlit.components.bbox_overlay import draw_face_bboxes
    from app.streamlit.components.image_popup import image_detail_btn

    st.subheader("Face Detection & Scoring")
    st.caption("Detected faces with bounding boxes and per-face scores")

    if not images:
        st.info("No image data available.")
        return

    images_with_faces = [img for img in images if img.face_count and img.face_count > 0]
    total_faces = sum(img.face_count for img in images_with_faces)

    col1, col2, col3 = st.columns(3)
    col1.metric("Images with Faces", len(images_with_faces))
    col2.metric("Total Faces", total_faces)
    col3.metric("Images without Faces", len(images) - len(images_with_faces))

    # Paginate
    page = st.session_state.get("explore_face_det_page", 0)
    total = len(images_with_faces)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    showing = images_with_faces[start:end]
    st.caption(f"Showing {start+1}-{end} of {total} images with faces")

    for img in showing:
        with st.container():
            cols = st.columns([2, 3])

            with cols[0]:
                # Draw image with face bounding boxes
                faces_for_bbox = []
                filter_scores = img.filter_scores or []
                for fs in filter_scores:
                    bbox = fs.get("bbox")
                    if bbox and (bbox.get("w_px", 0) > 0 or bbox.get("w", 0) > 0):
                        faces_for_bbox.append({
                            "bbox": bbox,
                            "label": f"face_{fs.get('face_index', 0)}",
                            "confidence": fs.get("confidence"),
                        })

                if faces_for_bbox:
                    annotated = draw_face_bboxes(img.path, faces_for_bbox, max_size=250)
                    if annotated:
                        st.image(annotated, width=250)
                    else:
                        thumb = _load_thumb(img.path, size=120)
                        if thumb:
                            st.image(thumb, width=120)
                else:
                    # No bbox data — show plain thumbnail
                    thumb = _load_thumb(img.path, size=120)
                    if thumb:
                        st.image(thumb, width=120)

                st.caption(f"{Path(img.path).name}")
                image_detail_btn(img, key=f"fdet_{Path(img.path).name}")

            with cols[1]:
                st.write(f"**{img.face_count} face(s)**")

                # Per-face scores
                pose_scores = img.face_pose_scores or []
                eyes_scores = img.face_eyes_scores or []
                smile_scores = img.face_smile_scores or []

                for i in range(img.face_count):
                    parts = [f"**Face #{i}**"]

                    # Detection confidence from filter_scores
                    if i < len(filter_scores):
                        conf = filter_scores[i].get("confidence")
                        if conf is not None:
                            parts.append(f"det:{conf:.2f}")

                    if i < len(pose_scores):
                        v = pose_scores[i]
                        color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
                        parts.append(f"pose::{color}[{v:.2f}]")
                    if i < len(eyes_scores):
                        v = eyes_scores[i]
                        color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
                        parts.append(f"eyes::{color}[{v:.2f}]")
                    if i < len(smile_scores):
                        v = smile_scores[i]
                        color = "green" if v >= 0.6 else ("orange" if v >= 0.3 else "red")
                        parts.append(f"expr::{color}[{v:.2f}]")

                    st.write(" | ".join(parts))

            st.divider()

    # Pagination
    if total > PAGE_SIZE:
        pcols = st.columns(3)
        with pcols[0]:
            if page > 0 and st.button("Previous", key="fdet_prev"):
                st.session_state["explore_face_det_page"] = page - 1
                st.rerun()
        with pcols[1]:
            st.caption(f"Page {page+1}/{(total-1)//PAGE_SIZE+1}")
        with pcols[2]:
            if end < total and st.button("Next", key="fdet_next"):
                st.session_state["explore_face_det_page"] = page + 1
                st.rerun()


def _render_scene_clustering_tab(all_decisions, job_id, img_lookup, images):
    """Scene clustering with visual image grids per cluster."""
    st.subheader("Scene Clustering")
    st.caption("Images grouped by visual similarity — which were selected from each group?")

    client = get_client()
    clusters = client.get_clusters(job_id)

    if not clusters:
        st.info("No cluster data available.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", len(clusters))
    col2.metric("Total Images", sum(c.image_count for c in clusters))
    col3.metric("Total Selected", sum(c.selected_count for c in clusters))

    # Show each cluster with images
    sorted_clusters = sorted(clusters, key=lambda c: c.image_count, reverse=True)

    for c in sorted_clusters:
        sel_count = c.selected_count
        total = c.image_count
        label = f"Cluster {c.cluster_id} — {total} images, {sel_count} selected"
        if c.has_faces:
            label += f" | {c.face_count or 0} faces"

        with st.expander(label, expanded=(total <= 10)):
            # Show images in this cluster as a grid with selected/rejected badges
            cluster_images = c.images if c.images else []
            if not cluster_images:
                st.caption("No image data available for this cluster")
                continue

            grid_cols = st.columns(min(len(cluster_images), 5))
            for i, img in enumerate(cluster_images[:20]):  # cap at 20 per cluster
                with grid_cols[i % 5]:
                    img_path = img.path if hasattr(img, 'path') else str(img)
                    thumb = _load_thumb(img_path, size=100)
                    if thumb:
                        st.image(thumb, width=100)
                    fname = Path(img_path).name if isinstance(img_path, str) else "?"
                    is_sel = img.is_selected if hasattr(img, 'is_selected') else False
                    score = img.composite_score if hasattr(img, 'composite_score') else None

                    if is_sel:
                        st.caption(f":green[Selected] {score:.2f}" if score else ":green[Selected]")
                    else:
                        st.caption(f":red[Rejected] {score:.2f}" if score else fname[:15])

            if len(cluster_images) > 20:
                st.caption(f"... and {len(cluster_images) - 20} more images")


def _render_face_clustering_tab(all_decisions, job_id, result, album):
    st.subheader("Face Clustering")

    fc_dir = result.get("fc_export_dir")
    if fc_dir:
        fc_url = f"http://localhost:8502/?load_run={fc_dir}"
        st.info(f"[Open in Face Clustering App]({fc_url}) -- analyze merges, recluster, train ML model\n\n`{fc_dir}`")

    client = get_client()
    people = client.get_people(album.album_id)
    if people:
        col1, col2, col3 = st.columns(3)
        col1.metric("People", len(people))
        col2.metric("Named", len([p for p in people if p.name]))
        col3.metric("Total Faces", sum(p.face_count for p in people))

        rows = [{"Name": p.name or f"Person {p.person_index + 1}",
                 "Faces": p.face_count, "Images": p.image_count} for p in people]
        st.dataframe(pd.DataFrame(rows).sort_values("Faces", ascending=False), use_container_width=True)

    face_decisions = [d for d in all_decisions if d.get("step") == "cluster_people"]
    if face_decisions:
        st.divider()
        counts = {}
        for d in face_decisions:
            counts[d.get("decision", "?")] = counts.get(d.get("decision", "?"), 0) + 1
        cols = st.columns(min(len(counts), 6))
        for i, (dec, count) in enumerate(sorted(counts.items())):
            if i < len(cols):
                cols[i].metric(dec.replace("_", " ").title(), count)


def _render_face_distance_tab(job_id: str, images: list) -> None:
    """Face distance calculator — pick two faces, see cosine distance."""
    st.subheader("Face Distance Calculator")
    st.caption("Select two faces from any images to compute their embedding distance")

    if not images:
        st.info("No image data available.")
        return

    images_with_faces = [img for img in images if img.face_count and img.face_count > 0]
    if not images_with_faces:
        st.info("No images with detected faces.")
        return

    img_names = [Path(img.path).name for img in images_with_faces]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Face A**")
        sel_a = st.selectbox("Image A", img_names, key="fd_img_a")
        img_a = next((img for img in images_with_faces if Path(img.path).name == sel_a), None)
        if img_a:
            thumb_a = _load_thumb(img_a.path, size=120)
            if thumb_a:
                st.image(thumb_a, width=120)
            face_opts_a = [f"Face #{i}" for i in range(img_a.face_count)]
            face_a = st.selectbox("Select face", face_opts_a, key="fd_face_a")
            face_idx_a = int(face_a.split("#")[1]) if face_a else 0

    with col2:
        st.markdown("**Face B**")
        sel_b = st.selectbox("Image B", img_names, key="fd_img_b", index=min(1, len(img_names)-1))
        img_b = next((img for img in images_with_faces if Path(img.path).name == sel_b), None)
        if img_b:
            thumb_b = _load_thumb(img_b.path, size=120)
            if thumb_b:
                st.image(thumb_b, width=120)
            face_opts_b = [f"Face #{i}" for i in range(img_b.face_count)]
            face_b = st.selectbox("Select face", face_opts_b, key="fd_face_b")
            face_idx_b = int(face_b.split("#")[1]) if face_b else 0

    st.divider()

    if st.button("Compute Distance", type="primary"):
        if img_a and img_b:
            try:
                client = get_client()
                resp = client._get(
                    f"/api/v1/results/{job_id}/face-distance",
                    params={
                        "image_path_a": img_a.path,
                        "face_index_a": face_idx_a,
                        "image_path_b": img_b.path,
                        "face_index_b": face_idx_b,
                    }
                )

                dist = resp.get("cosine_distance", -1)
                verdict = resp.get("verdict", "unknown")

                # Display result
                col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
                with col_r1:
                    st.metric("Distance", f"{dist:.4f}")
                with col_r2:
                    if verdict == "same_person":
                        st.success(f"Same Person (distance {dist:.3f} < 0.3)")
                    elif verdict == "borderline":
                        st.warning(f"Borderline (distance {dist:.3f}, between 0.3-0.5)")
                    else:
                        st.error(f"Different People (distance {dist:.3f} > 0.5)")
                with col_r3:
                    st.metric("Verdict", verdict.replace("_", " ").title())

            except Exception as e:
                st.error(f"Failed to compute distance: {e}")


def _find_image(item_id: str, img_lookup: Dict):
    """Find ImageInfo by item_id (which might be a path or face key)."""
    # Direct match
    if item_id in img_lookup:
        return img_lookup[item_id]
    # Try matching by filename
    name = Path(item_id).name
    for path, info in img_lookup.items():
        if Path(path).name == name:
            return info
    # Face key format: "path:face_N"
    if ":face_" in item_id:
        img_path = item_id.rsplit(":face_", 1)[0]
        return img_lookup.get(img_path)
    return None
