"""Enlarged face image with landmark and pose overlay."""

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app.face_clustering_debug.models.schemas import FaceInfo

# Landmark colors and labels: left_eye, right_eye, nose, left_mouth, right_mouth
_LANDMARK_INFO = [
    ("LE", "red"),      # Left eye
    ("RE", "red"),      # Right eye
    ("N", "green"),     # Nose
    ("LM", "blue"),     # Left mouth corner
    ("RM", "blue"),     # Right mouth corner
]

# ArcFace reference template landmarks (normalized to 0-1 for 112x112 base)
# These are where landmarks END UP after 5-point alignment
_ARCFACE_REF_LANDMARKS_NORMALIZED: List[Tuple[float, float]] = [
    (38.2946 / 112.0, 51.6963 / 112.0),   # left eye: (0.342, 0.461)
    (73.5318 / 112.0, 51.5014 / 112.0),   # right eye: (0.657, 0.460)
    (56.0252 / 112.0, 71.7366 / 112.0),   # nose: (0.500, 0.640)
    (41.5493 / 112.0, 92.3655 / 112.0),   # left mouth: (0.371, 0.825)
    (70.7299 / 112.0, 92.2041 / 112.0),   # right mouth: (0.632, 0.823)
]


def render_face_detail(
    face: FaceInfo,
    crop_bytes: bytes,
    use_aligned_landmarks: bool = True,
) -> None:
    """Render an enlarged face with landmarks, filename, and metadata.

    Args:
        face: Face metadata.
        crop_bytes: JPEG bytes of the face crop.
        use_aligned_landmarks: If True (default), use ArcFace reference template
            positions for landmarks since face crops are 5-point aligned.
            If False, use original landmarks from face metadata.
    """
    img = Image.open(BytesIO(crop_bytes))

    # Draw landmarks with labels
    # Since face crops are 5-point aligned, landmarks are at reference template positions
    if use_aligned_landmarks:
        img = _draw_landmarks_with_labels(img, _ARCFACE_REF_LANDMARKS_NORMALIZED)
    elif face.landmarks and len(face.landmarks) == 5:
        img = _draw_landmarks_with_labels(img, face.landmarks)

    # Extract filename
    filename = Path(face.image_path).name if face.image_path else "unknown"
    st.image(img, caption=f"Face #{face.index} — {filename}", use_container_width=True)

    # File info section
    st.markdown("**File Info**")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text(f"File: {filename}")
        if face.image_path:
            st.text(f"Path: {face.image_path}")
    with col2:
        if face.image_path:
            st.code(face.image_path, language=None)

    # Face metrics section
    st.markdown("**Face Metrics**")
    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.text(f"Confidence: {face.confidence:.3f}")
        st.text(f"Frontal Score: {face.frontal_score:.3f}")
        if face.eye_bbox_ratio > 0:
            st.text(f"Eye-Bbox Ratio: {face.eye_bbox_ratio:.3f}")

    with metrics_col2:
        # Pose angles
        if face.pose_angles:
            pitch, yaw, roll = face.pose_angles
            st.text(f"Pitch: {pitch:.1f}°")
            st.text(f"Yaw: {yaw:.1f}°")
            st.text(f"Roll: {roll:.1f}°")

    # Bbox info
    if face.bbox:
        x, y, w, h = face.bbox
        st.text(f"Bbox: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})")

    # Landmark legend
    with st.expander("Landmark Legend"):
        st.markdown("""
        - **LE** (red): Left eye
        - **RE** (red): Right eye
        - **N** (green): Nose tip
        - **LM** (blue): Left mouth corner
        - **RM** (blue): Right mouth corner
        """)


def render_face_debug_panel(
    face: FaceInfo,
    aligned_crop: Optional[bytes],
    raw_crop: Optional[bytes],
    original_with_bbox: Optional[bytes],
) -> None:
    """Render comprehensive debug panel showing all three face versions.

    Args:
        face: Face metadata.
        aligned_crop: JPEG bytes of 5-point aligned face crop.
        raw_crop: JPEG bytes of raw bbox crop (no alignment).
        original_with_bbox: JPEG bytes of original image with bbox/landmarks drawn.
    """
    filename = Path(face.image_path).name if face.image_path else "unknown"
    st.markdown(f"### Debug Panel: Face #{face.index} — `{filename}`")

    # Three columns for the three versions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Original + BBox**")
        st.caption("Source image with detected bbox (green) and landmarks (colored dots)")
        if original_with_bbox:
            st.image(original_with_bbox, use_container_width=True)
        else:
            st.warning("Original image not available")

    with col2:
        st.markdown("**2. Raw Crop**")
        st.caption("Bbox crop only, no rotation/alignment")
        if raw_crop:
            img_raw = Image.open(BytesIO(raw_crop))
            # Draw original landmarks if available (normalized to crop)
            if face.landmarks and len(face.landmarks) == 5:
                img_raw = _draw_landmarks_with_labels(img_raw, face.landmarks)
            st.image(img_raw, use_container_width=True)
        else:
            st.warning("Raw crop not available")

    with col3:
        st.markdown("**3. Aligned Crop**")
        st.caption("5-point affine alignment to ArcFace template")
        if aligned_crop:
            img_aligned = Image.open(BytesIO(aligned_crop))
            # Draw reference template landmarks
            img_aligned = _draw_landmarks_with_labels(img_aligned, _ARCFACE_REF_LANDMARKS_NORMALIZED)
            st.image(img_aligned, use_container_width=True)
        else:
            st.warning("Aligned crop not available")

    # Metadata section
    with st.expander("Face Metadata", expanded=False):
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown("**Detection**")
            st.text(f"Confidence: {face.confidence:.3f}")
            st.text(f"Frontal Score: {face.frontal_score:.3f}")
            if face.bbox:
                x, y, w, h = face.bbox
                st.text(f"Bbox: ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")

        with meta_col2:
            st.markdown("**Pose**")
            if face.pose_angles:
                pitch, yaw, roll = face.pose_angles
                st.text(f"Pitch: {pitch:.1f}°")
                st.text(f"Yaw: {yaw:.1f}°")
                st.text(f"Roll: {roll:.1f}°")
            else:
                st.text("Pose angles not available")

        st.markdown("**File**")
        st.code(face.image_path or "unknown", language=None)


def _draw_landmarks_with_labels(img: Image.Image, landmarks: list) -> Image.Image:
    """Return copy of image with 5-point landmarks and labels drawn."""
    result = img.copy()
    draw = ImageDraw.Draw(result)
    w, h = result.size
    r = max(3, int(min(w, h) * 0.03))

    for (x, y), (label, color) in zip(landmarks, _LANDMARK_INFO):
        px, py = x * w, y * h

        # Draw circle
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=1)

        # Draw label above the point
        label_x = px - r
        label_y = py - r * 3
        draw.text((label_x, label_y), label, fill="white")

    return result
