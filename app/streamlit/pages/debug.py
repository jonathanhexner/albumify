"""Debug page for face analysis and pipeline inspection."""

import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from app.streamlit.session import get_session
from app.streamlit.api_client import get_client

logger = logging.getLogger(__name__)


def _image_to_base64(image_path: Path, size: int = 100) -> Optional[str]:
    """Load image and return base64 data URI."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def _crop_face(image_path: Path, bbox: Dict, padding: float = 0.2) -> Optional[Image.Image]:
    """Crop face from image with padding."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            x, y = bbox.get("x_px", 0), bbox.get("y_px", 0)
            w, h = bbox.get("w_px", 0), bbox.get("h_px", 0)

            if w <= 0 or h <= 0:
                return None

            pad = int(min(w, h) * padding)
            left = max(0, x - pad)
            top = max(0, y - pad)
            right = min(img.width, x + w + pad)
            bottom = min(img.height, y + h + pad)

            return img.crop((left, top, right, bottom))
    except Exception as e:
        logger.warning(f"Failed to crop face from {image_path}: {e}")
        return None


def _face_crop_to_base64(face_crop: Image.Image, size: int = 80) -> str:
    """Convert face crop to base64."""
    face_crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    face_crop.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_filtering_explanation():
    """Render explanation of the filtering process."""
    st.markdown("""
    ## Face Filtering Pipeline

    The pipeline applies **two-stage filtering** to ensure only high-quality, frontal faces are used for clustering:

    ### Stage 1: Basic Quality Filtering (`filter_faces` step)
    Removes faces that are too small or unreliable:

    | Filter | Threshold | Description |
    |--------|-----------|-------------|
    | **Confidence** | >= 0.5 | Detection confidence from InsightFace |
    | **BBox Ratio** | >= 0.02 | Face width / image width (removes tiny background faces) |
    | **Relative Size** | >= 0.3 | Face width / largest face in image (removes small secondary faces) |
    | **Eye Ratio** | >= 0.01 | Inter-eye distance / image width |

    ### Stage 2: Frontal Scoring (`score_face_frontal` step)
    Computes frontal score and marks non-frontal faces as non-clusterable:

    | Metric | Ideal | Profile | Description |
    |--------|-------|---------|-------------|
    | **Eye/BBox Ratio** | >= 0.25 | < 0.15 | Inter-eye distance / bbox width |
    | **Asymmetry** | ~1.0 | > 2.0 | Nose-to-eye distance ratio |
    | **Frontal Score** | >= 0.4 | < 0.4 | Combined score (clusterable threshold) |

    ### Result
    - Faces with `filter_passed=False` are ignored completely
    - Faces with `is_clusterable=False` don't get embeddings (not in People tab)
    - Non-frontal faces still contribute to **selection penalty** (prefer images with frontal faces)
    """)


def render_config_knobs():
    """Render configuration sliders."""
    st.subheader("Filtering Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Stage 1: Basic Filters**")
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05,
                                   help="Detection confidence threshold")
        min_bbox_ratio = st.slider("Min BBox Ratio", 0.0, 0.2, 0.02, 0.005,
                                   help="Min face_width / image_width")
        min_relative_size = st.slider("Min Relative Size", 0.0, 1.0, 0.3, 0.05,
                                      help="Min face_width / max_face_width in image")

    with col2:
        st.markdown("**Stage 2: Frontal Filters**")
        min_frontal_score = st.slider("Min Frontal Score", 0.0, 1.0, 0.4, 0.05,
                                      help="Threshold for is_clusterable")
        min_eye_bbox_ratio = st.slider("Min Eye/BBox Ratio", 0.0, 0.4, 0.20, 0.02,
                                       help="Inter-eye / bbox width")
        max_asymmetry = st.slider("Max Asymmetry", 1.0, 3.0, 1.8, 0.1,
                                  help="Nose-eye distance ratio threshold")

    return {
        "min_confidence": min_confidence,
        "min_bbox_ratio": min_bbox_ratio,
        "min_relative_size": min_relative_size,
        "min_frontal_score": min_frontal_score,
        "min_eye_bbox_ratio": min_eye_bbox_ratio,
        "max_asymmetry": max_asymmetry,
    }


def get_face_data_from_api(album_id: str, run_id: str) -> Dict[str, Any]:
    """Fetch face data from API."""
    client = get_client()

    # Get images with metrics
    images = client.get_images(run_id)

    # Get people/clusters
    people = client.get_people(album_id, run_id)

    return {
        "images": images,
        "people": people,
    }


def render_face_scores_table(images: List, source_dir: Path):
    """Render table of face scores for all images."""
    st.subheader("Face Scores Table")

    rows = []
    for img in images:
        img_path = Path(img.path)

        # Get filter scores
        filter_scores = getattr(img, 'filter_scores', None) or []
        frontal_scores = getattr(img, 'frontal_scores', None) or []

        if not filter_scores and not frontal_scores:
            # No face data - add placeholder row
            rows.append({
                "Image": img_path.name,
                "Face": "-",
                "Confidence": "-",
                "BBox Ratio": "-",
                "Rel Size": "-",
                "Filter": "-",
                "Frontal": "-",
                "Eye/BBox": "-",
                "Asymmetry": "-",
                "Cluster": "-",
            })
            continue

        # Merge filter and frontal scores by face_index
        face_data = {}
        for fs in filter_scores:
            idx = fs.get('face_index', 0)
            face_data[idx] = {'filter': fs}
        for fs in frontal_scores:
            idx = fs.get('face_index', 0)
            if idx in face_data:
                face_data[idx]['frontal'] = fs
            else:
                face_data[idx] = {'frontal': fs}

        for face_idx, data in face_data.items():
            f_scores = data.get('filter', {})
            fr_scores = data.get('frontal', {})

            rows.append({
                "Image": img_path.name,
                "Face": f"#{face_idx}",
                "Confidence": f"{f_scores.get('confidence', 0):.2f}" if f_scores.get('confidence') else "-",
                "BBox Ratio": f"{f_scores.get('bbox_ratio', 0):.3f}" if f_scores.get('bbox_ratio') else "-",
                "Rel Size": f"{f_scores.get('relative_size', 0):.2f}" if f_scores.get('relative_size') else "-",
                "Filter": "Pass" if f_scores.get('filter_passed', True) else "FAIL",
                "Frontal": f"{fr_scores.get('frontal_score', 0):.2f}" if fr_scores.get('frontal_score') else "-",
                "Eye/BBox": f"{fr_scores.get('eye_bbox_ratio', 0):.3f}" if fr_scores.get('eye_bbox_ratio') else "-",
                "Asymmetry": f"{fr_scores.get('asymmetry', 0):.2f}" if fr_scores.get('asymmetry') else "-",
                "Cluster": "Yes" if fr_scores.get('is_clusterable', True) else "NO",
            })

    if not rows:
        st.info("No face data available. Run the pipeline first.")
        return

    df = pd.DataFrame(rows)

    # Color coding
    def highlight_failures(row):
        styles = [''] * len(row)
        if row['Filter'] == 'FAIL':
            styles[5] = 'background-color: #ffcdd2'
        if row['Cluster'] == 'NO':
            styles[9] = 'background-color: #fff3e0'
        return styles

    styled_df = df.style.apply(highlight_failures, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Summary stats
    total_faces = len([r for r in rows if r['Face'] != '-'])
    passed_filter = len([r for r in rows if r['Filter'] == 'Pass'])
    clusterable = len([r for r in rows if r['Cluster'] == 'Yes'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Faces", total_faces)
    col2.metric("Passed Filter", passed_filter, f"{passed_filter - total_faces}" if passed_filter < total_faces else None)
    col3.metric("Clusterable", clusterable, f"{clusterable - passed_filter}" if clusterable < passed_filter else None)


def render_face_distance_matrix(images: List, source_dir: Path):
    """Render face embedding distance matrix."""
    st.subheader("Face Distance Matrix")

    st.info("Distance matrix shows cosine similarity between face embeddings. "
            "Lower values = more similar. Faces of the same person should have low distances.")

    # This would require fetching embeddings from API
    # For now, show placeholder
    st.warning("Face embedding distances require API endpoint. Coming soon.")


def render_image_detail_view(images: List, source_dir: Path):
    """Render detailed view for selected image."""
    st.subheader("Image Detail View")

    if not images:
        st.info("No images available.")
        return

    # Image selector
    image_names = [Path(img.path).name for img in images]
    selected_name = st.selectbox("Select Image", image_names)

    selected_img = next((img for img in images if Path(img.path).name == selected_name), None)
    if not selected_img:
        return

    img_path = Path(selected_img.path)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Show image with face boxes
        try:
            with Image.open(img_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                draw = ImageDraw.Draw(pil_img)

                # Draw face boxes from filter_scores
                filter_scores = getattr(selected_img, 'filter_scores', None) or []
                frontal_scores = getattr(selected_img, 'frontal_scores', None) or []

                # We don't have bbox in filter_scores currently
                # This would need API enhancement

                # Resize for display
                pil_img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                st.image(pil_img, caption=selected_name)
        except Exception as e:
            st.error(f"Failed to load image: {e}")

    with col2:
        st.markdown("**Face Metrics**")

        filter_scores = getattr(selected_img, 'filter_scores', None) or []
        frontal_scores = getattr(selected_img, 'frontal_scores', None) or []

        if filter_scores:
            for fs in filter_scores:
                face_idx = fs.get('face_index', 0)
                st.markdown(f"**Face #{face_idx}**")
                st.write(f"- Confidence: {fs.get('confidence', 'N/A')}")
                st.write(f"- BBox Ratio: {fs.get('bbox_ratio', 'N/A')}")
                st.write(f"- Relative Size: {fs.get('relative_size', 'N/A')}")
                st.write(f"- Filter Passed: {'Yes' if fs.get('filter_passed', True) else 'NO'}")

                # Find matching frontal score
                fr = next((f for f in frontal_scores if f.get('face_index') == face_idx), {})
                if fr:
                    st.write(f"- Frontal Score: {fr.get('frontal_score', 'N/A')}")
                    st.write(f"- Eye/BBox Ratio: {fr.get('eye_bbox_ratio', 'N/A')}")
                    st.write(f"- Asymmetry: {fr.get('asymmetry', 'N/A')}")
                    st.write(f"- Clusterable: {'Yes' if fr.get('is_clusterable', True) else 'NO'}")
                st.markdown("---")
        else:
            st.info("No face metrics available for this image.")

        # Additional image metrics
        st.markdown("**Image Metrics**")
        st.write(f"- IQA Score: {selected_img.iqa_score or 'N/A'}")
        st.write(f"- AVA Score: {selected_img.ava_score or 'N/A'}")
        st.write(f"- Composite Score: {selected_img.composite_score or 'N/A'}")
        st.write(f"- Face Count: {selected_img.face_count}")
        st.write(f"- Selected: {'Yes' if selected_img.is_selected else 'No'}")


def render_debug_page():
    """Main debug page rendering."""
    st.title("Debug: Face Analysis")

    session = get_session()

    if not session.api_connected:
        st.error("API not connected. Please check the backend server.")
        return

    if not session.current_album:
        st.warning("No album selected. Please select an album first.")
        return

    album = session.current_album
    source_dir = Path(album.source_directory)

    st.info(f"Analyzing album: **{album.name}** ({album.total_images} images)")

    # Check for pipeline results
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.warning("No pipeline results found. Run the pipeline first.")
        return

    # Use most recent result
    latest_result = results[0]
    run_id = latest_result.get("job_id")

    st.caption(f"Using pipeline run: {run_id[:8]}...")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Filtering Explanation",
        "Face Scores Table",
        "Image Detail",
        "Config Knobs"
    ])

    # Get images
    images = client.get_images(run_id)

    with tab1:
        render_filtering_explanation()

    with tab2:
        render_face_scores_table(images, source_dir)

    with tab3:
        render_image_detail_view(images, source_dir)

    with tab4:
        config = render_config_knobs()
        st.markdown("---")
        st.markdown("**Note**: These sliders show current thresholds but don't modify the pipeline config yet. "
                   "To change thresholds, edit `configs/pipeline.yaml` and restart the API.")

        if st.button("Apply Config Changes (Coming Soon)"):
            st.info("Config modification via UI is not yet implemented.")
