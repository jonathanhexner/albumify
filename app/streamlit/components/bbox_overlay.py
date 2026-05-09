"""Bounding box overlay utility for face displays."""

import io
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)

# Colors for face bounding boxes
FACE_COLORS = ["#4ecca3", "#e94560", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71"]


def draw_face_bboxes(
    image_path: str,
    faces: List[Dict],
    highlight_index: Optional[int] = None,
    max_size: int = 400,
) -> Optional[Image.Image]:
    """Draw face bounding boxes on an image.

    Args:
        image_path: Path to source image
        faces: List of face dicts with 'bbox' key containing {x_px, y_px, w_px, h_px}
        highlight_index: Index of the face to highlight (green, thicker border)
        max_size: Max dimension for output image

    Returns:
        Annotated PIL Image, or None on failure
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            img_w, img_h = img.size

            # Collect bbox coordinates in normalized form
            face_boxes = []
            for i, face in enumerate(faces):
                bbox = face.get("bbox", {})

                if bbox.get("x_px") is not None and int(bbox.get("w_px", 0)) > 0:
                    nx = bbox["x_px"] / img_w
                    ny = bbox["y_px"] / img_h
                    nw = bbox["w_px"] / img_w
                    nh = bbox["h_px"] / img_h
                elif bbox.get("x") is not None and bbox.get("w", 0) > 0:
                    nx, ny, nw, nh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                else:
                    continue
                face_boxes.append((i, face, nx, ny, nw, nh))

            # Resize FIRST, then draw bboxes (so line widths are visible)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            out_w, out_h = img.size
            draw = ImageDraw.Draw(img)

            for i, face, nx, ny, nw, nh in face_boxes:
                x = int(nx * out_w)
                y = int(ny * out_h)
                w = int(nw * out_w)
                h = int(nh * out_h)

                if w <= 0 or h <= 0:
                    continue

                is_highlight = (i == highlight_index)
                color = "#4ecca3" if is_highlight else FACE_COLORS[i % len(FACE_COLORS)]
                width = 3 if is_highlight else 2

                draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

                label = face.get("label", f"face_{i}")
                conf = face.get("confidence", None)
                text = label
                if conf is not None:
                    text += f" ({conf:.2f})"
                draw.text((x, max(0, y - 14)), text, fill=color)

            return img
    except Exception as e:
        logger.warning(f"Failed to draw bboxes on {image_path}: {e}")
        return None


def image_with_bboxes_to_base64(
    image_path: str,
    faces: List[Dict],
    highlight_index: Optional[int] = None,
    max_size: int = 400,
) -> Optional[str]:
    """Draw bboxes and return as base64 data URI."""
    img = draw_face_bboxes(image_path, faces, highlight_index, max_size)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_thumbnail(image_path: str, size: int = 100) -> Optional[str]:
    """Load image thumbnail as base64 data URI."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            sq = min(img.size)
            left = (img.width - sq) // 2
            top = (img.height - sq) // 2
            img = img.crop((left, top, left + sq, top + sq))
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None
