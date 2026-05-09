"""
Benchmark MediaPipe vs InsightFace face alignment.

Compares rotation angles computed from landmarks and generates aligned face crops.
Creates HTML report with side-by-side comparison.

Usage:
    python scripts/benchmark_face_alignment.py --dataset-dir path/to/images
"""

import argparse
import base64
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from sim_bench.pipeline.insightface_pipeline.face_analyzer import InsightFaceFaceAnalyzer
from sim_bench.pipeline.utils.image_cache import get_image_cache
from sim_bench.pipeline.utils.face_alignment import rotate_image_and_transform_bbox, crop_aligned_face

logger = logging.getLogger(__name__)


def compute_roll_angle(landmarks: List[List[float]]) -> float:
    """Compute roll angle from eye landmarks.

    Returns angle in degrees. Positive = clockwise tilt.
    """
    if not landmarks or len(landmarks) < 2:
        return 0.0

    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]

    roll_angle = math.atan2(dy, dx) * 180 / math.pi
    return roll_angle


def extract_mediapipe_landmarks(image: np.ndarray, face_mesh) -> Optional[Dict[str, np.ndarray]]:
    """Extract landmarks from image using MediaPipe (468+ points available).

    Args:
        image: RGB image as numpy array
        face_mesh: Pre-initialised MediaPipe FaceMesh instance (reused across calls for performance)

    Returns:
        Dict with:
        - 'all_landmarks': All available landmarks as (N, 2) array in pixel coordinates
        - 'key_landmarks': Key 5-point landmarks [left_eye, right_eye, nose, mouth_left, mouth_right] 
          matching InsightFace format, as (5, 2) array
        - 'eye_landmarks': Eye centers for rotation calculation, as (2, 2) array
        Returns None if no face detected
    """
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]

    # Extract ALL available landmarks (468 or 478 with refine_landmarks=True)
    all_landmarks = []
    for lm in landmarks.landmark:
        all_landmarks.append([lm.x * w, lm.y * h])
    all_landmarks_array = np.array(all_landmarks, dtype=np.float32)

    # Extract key 5-point landmarks matching InsightFace format
    # left_eye, right_eye, nose, mouth_left, mouth_right
    key_indices = [468, 473, 4, 61, 291]  # Left eye, right eye, nose, mouth left, mouth right
    
    key_landmarks = []
    for idx in key_indices:
        if idx >= len(landmarks.landmark):
            return None
        lm = landmarks.landmark[idx]
        key_landmarks.append([lm.x * w, lm.y * h])
    key_landmarks_array = np.array(key_landmarks, dtype=np.float32)

    # Extract eye centers for rotation calculation
    # Use .copy() to avoid numpy view aliasing: without it, modifying key_landmarks also modifies eye_landmarks
    eye_landmarks_array = key_landmarks_array[:2].copy()

    return {
        'all_landmarks': all_landmarks_array,
        'key_landmarks': key_landmarks_array,
        'eye_landmarks': eye_landmarks_array
    }




def overlay_landmarks_on_rotated_crop(
    face_crop: np.ndarray,
    landmarks_original: np.ndarray,
    bbox_expanded_original: Dict[str, Any],
    bbox_expanded_rotated: Dict[str, Any],
    original_image_shape: Tuple[int, int],
    rotation_angle: float,
    target_size: int,
    key_landmarks: Optional[np.ndarray] = None,
    show_all: bool = False
) -> np.ndarray:
    """Overlay landmarks on rotated and cropped face.

    Args:
        face_crop: Cropped face image (target_size x target_size)
        landmarks_original: Landmarks in original image coordinates (N, 2) array
            - For InsightFace: (5, 2) - key landmarks only
            - For MediaPipe: can be all landmarks (468+) if show_all=True
        bbox_expanded_original: Expanded bbox in original image
        bbox_expanded_rotated: Expanded bbox in rotated image
        original_image_shape: Shape of original image (h, w)
        rotation_angle: Rotation angle applied (degrees, negative of roll)
        target_size: Target size of crop
        key_landmarks: Optional key landmarks (5, 2) for highlighting (e.g., eyes, nose, mouth)
        show_all: If True, show all landmarks (for MediaPipe with 468+ points)

    Returns:
        Face crop with landmarks overlaid
    """
    face_with_landmarks = face_crop.copy()
    h_crop, w_crop = face_crop.shape[:2]
    
    # Use the same rotation logic as rotate_image_and_transform_bbox
    h_orig, w_orig = original_image_shape
    center = (w_orig // 2, h_orig // 2)
    
    # Calculate new image size after rotation (same as in rotate_image_and_transform_bbox)
    angle_rad = math.radians(abs(rotation_angle))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_w = int((h_orig * sin_a) + (w_orig * cos_a))
    new_h = int((h_orig * cos_a) + (w_orig * sin_a))
    
    # Create rotation matrix (same as in rotate_image_and_transform_bbox)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Transform landmarks to rotated image coordinates
    landmarks_rotated = cv2.transform(
        landmarks_original.reshape(-1, 1, 2).astype(np.float32),
        rotation_matrix
    ).reshape(-1, 2)
    
    # Transform key landmarks if provided
    key_landmarks_rotated = None
    if key_landmarks is not None:
        key_landmarks_rotated = cv2.transform(
            key_landmarks.reshape(-1, 1, 2).astype(np.float32),
            rotation_matrix
        ).reshape(-1, 2)
    
    # Transform to crop coordinates
    crop_x = bbox_expanded_rotated['x_px']
    crop_y = bbox_expanded_rotated['y_px']
    crop_w = bbox_expanded_rotated['w_px']
    crop_h = bbox_expanded_rotated['h_px']
    
    # Scale factor from crop size to target_size
    scale_x = w_crop / crop_w if crop_w > 0 else 1.0
    scale_y = h_crop / crop_h if crop_h > 0 else 1.0
    
    # Draw all landmarks (small points)
    if show_all:
        for landmark in landmarks_rotated:
            lx = (landmark[0] - crop_x) * scale_x
            ly = (landmark[1] - crop_y) * scale_y
            
            # Only draw if within crop bounds
            if 0 <= lx < w_crop and 0 <= ly < h_crop:
                x_int = int(lx)
                y_int = int(ly)
                # Small gray dots for all landmarks
                cv2.circle(face_with_landmarks, (x_int, y_int), 1, (128, 128, 128), -1)
    
    # Draw key landmarks (larger, colored points with labels)
    landmarks_to_draw = key_landmarks_rotated if key_landmarks_rotated is not None else landmarks_rotated
    landmark_labels = ['L-Eye', 'R-Eye', 'Nose', 'L-Mouth', 'R-Mouth']
    
    for i, landmark in enumerate(landmarks_to_draw):
        # Transform to crop-relative coordinates
        lx = (landmark[0] - crop_x) * scale_x
        ly = (landmark[1] - crop_y) * scale_y
        
        # Only draw if within crop bounds (with small margin)
        if -5 <= lx < w_crop + 5 and -5 <= ly < h_crop + 5:
            # Green for eyes, red for nose, blue for mouth
            if i < 2:
                color = (0, 255, 0)  # Eyes - green
            elif i == 2:
                color = (255, 0, 0)  # Nose - red
            else:
                color = (0, 0, 255)  # Mouth - blue
            
            # Draw landmark point (larger for key landmarks)
            x_int = int(max(0, min(w_crop - 1, lx)))
            y_int = int(max(0, min(h_crop - 1, ly)))
            cv2.circle(face_with_landmarks, (x_int, y_int), 4, color, -1)
            cv2.circle(face_with_landmarks, (x_int, y_int), 6, (255, 255, 255), 2)
            
            # Add label
            if i < len(landmark_labels):
                cv2.putText(face_with_landmarks, landmark_labels[i], 
                           (x_int + 8, y_int - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.3, (255, 255, 255), 1)
    
    return face_with_landmarks


def image_to_base64(image: np.ndarray, max_size: int = 200) -> str:
    """Convert image to base64 string for HTML embedding.

    Args:
        image: Image as numpy array
        max_size: Maximum dimension for thumbnail

    Returns:
        Base64 encoded image string
    """
    # Resize if needed
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to PIL and encode
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return f"data:image/jpeg;base64,{img_base64}"


def process_image(
    image_path: Path,
    face_analyzer: InsightFaceFaceAnalyzer,
    output_dir: Path,
    margin: float,
    target_size: int,
    face_mesh=None
) -> Optional[Dict[str, Any]]:
    """Process single image: detect face, extract landmarks, align and crop.

    Returns:
        Dict with results or None if processing failed
    """
    cache = get_image_cache()
    
    try:
        # Load image
        image = cache.get(image_path)
        h, w = image.shape[:2]

        # Detect face with InsightFace
        detections = face_analyzer.detect_faces(image_path, None)
        if not detections:
            logger.warning(f"No face detected in {image_path.name}")
            return None

        # Use first face
        detection = detections[0]
        bbox_original = {
            'x_px': detection.bbox.x_px,
            'y_px': detection.bbox.y_px,
            'w_px': detection.bbox.w_px,
            'h_px': detection.bbox.h_px
        }

        # Get InsightFace landmarks (already in pixel coordinates)
        if_landmarks = detection.landmarks
        if_eye_landmarks = [[if_landmarks[0][0], if_landmarks[0][1]], 
                           [if_landmarks[1][0], if_landmarks[1][1]]]

        # Extract MediaPipe landmarks - try full image first, then expanded face crop
        mp_result = extract_mediapipe_landmarks(image, face_mesh)
        if mp_result is None:
            # Try extracting from expanded face crop (30% margin) for better MediaPipe detection
            h, w = image.shape[:2]
            margin_expand = 0.3
            margin_w = int(bbox_original['w_px'] * margin_expand)
            margin_h = int(bbox_original['h_px'] * margin_expand)
            x1 = max(0, bbox_original['x_px'] - margin_w)
            y1 = max(0, bbox_original['y_px'] - margin_h)
            x2 = min(w, bbox_original['x_px'] + bbox_original['w_px'] + margin_w)
            y2 = min(h, bbox_original['y_px'] + bbox_original['h_px'] + margin_h)
            
            if x2 > x1 and y2 > y1:
                face_crop = image[y1:y2, x1:x2]
                mp_result = extract_mediapipe_landmarks(face_crop, face_mesh)
                if mp_result is not None:
                    # Convert from crop pixel coordinates to full image coordinates
                    mp_result['all_landmarks'][:, 0] += x1
                    mp_result['all_landmarks'][:, 1] += y1
                    mp_result['key_landmarks'][:, 0] += x1
                    mp_result['key_landmarks'][:, 1] += y1
                    mp_result['eye_landmarks'][:, 0] += x1
                    mp_result['eye_landmarks'][:, 1] += y1
        
        # Compute InsightFace rotation angle
        angle_if = compute_roll_angle(if_eye_landmarks)

        # Handle MediaPipe failure - still process with InsightFace only
        mp_failed = mp_result is None
        if mp_failed:
            logger.warning(f"MediaPipe failed to detect landmarks in {image_path.name} (tried full image and face crop)")
            angle_mp = None
            delta = None
            mp_all_landmarks = None
            mp_key_landmarks = None
        else:
            # Use eye landmarks for rotation calculation
            mp_eye_landmarks = mp_result['eye_landmarks'].tolist()
            angle_mp = compute_roll_angle(mp_eye_landmarks)
            delta = abs(angle_if - angle_mp)
            mp_all_landmarks = mp_result['all_landmarks']
            mp_key_landmarks = mp_result['key_landmarks']

        # Create expanded bounding box (original + 30% margin) for rotation
        expand_margin = 0.3
        bbox_expanded = {
            'x_px': max(0, bbox_original['x_px'] - int(bbox_original['w_px'] * expand_margin)),
            'y_px': max(0, bbox_original['y_px'] - int(bbox_original['h_px'] * expand_margin)),
            'w_px': bbox_original['w_px'] + 2 * int(bbox_original['w_px'] * expand_margin),
            'h_px': bbox_original['h_px'] + 2 * int(bbox_original['h_px'] * expand_margin)
        }
        # Clamp to image bounds
        bbox_expanded['w_px'] = min(bbox_expanded['w_px'], w - bbox_expanded['x_px'])
        bbox_expanded['h_px'] = min(bbox_expanded['h_px'], h - bbox_expanded['y_px'])

        # Rotate and crop for InsightFace
        # Flow: 1) Expand bbox by 30%, 2) Rotate full image, 3) Transform expanded bbox, 4) Crop tighter bbox
        # Positive angle_if = clockwise tilt → rotate by +angle_if (CCW in OpenCV) to correct
        rotated_if, bbox_expanded_if = rotate_image_and_transform_bbox(image, bbox_expanded, angle_if)
        # Crop the expanded bbox (which contains the aligned face)
        face_if = crop_aligned_face(rotated_if, bbox_expanded_if, margin=margin, target_size=target_size)
        if face_if is None:
            logger.warning(f"Failed to crop InsightFace-aligned face from {image_path.name}")
            return None

        # Rotate and crop for MediaPipe (if landmarks were detected)
        if not mp_failed:
            rotated_mp, bbox_expanded_mp = rotate_image_and_transform_bbox(image, bbox_expanded, angle_mp)
            face_mp = crop_aligned_face(rotated_mp, bbox_expanded_mp, margin=margin, target_size=target_size)
            if face_mp is None:
                logger.warning(f"Failed to crop MediaPipe-aligned face from {image_path.name}")
                face_mp = None
                mp_failed = True
        else:
            face_mp = None

        # Overlay landmarks on face crops
        # NOTE: Landmarks are shown AFTER rotation correction, so eyes should be horizontal
        # We transform landmarks from original image -> rotated image -> cropped image
        # image.shape is (h, w, c), we need (h, w)
        image_shape_2d = (image.shape[0], image.shape[1])
        
        # Ensure InsightFace landmarks are numpy arrays with shape (5, 2)
        if_landmarks_array = np.array(if_landmarks, dtype=np.float32) if not isinstance(if_landmarks, np.ndarray) else if_landmarks.astype(np.float32)
        if if_landmarks_array.shape != (5, 2):
            logger.warning(f"InsightFace landmarks shape {if_landmarks_array.shape} != (5, 2) for {image_path.name}")
        
        # InsightFace: show only 5 key landmarks
        face_if_with_landmarks = overlay_landmarks_on_rotated_crop(
            face_if, if_landmarks_array, bbox_expanded, bbox_expanded_if, image_shape_2d, angle_if, target_size,
            key_landmarks=None, show_all=False
        )
        
        if not mp_failed and face_mp is not None:
            # MediaPipe: show ALL landmarks (468+) with key landmarks highlighted
            face_mp_with_landmarks = overlay_landmarks_on_rotated_crop(
                face_mp, mp_all_landmarks, bbox_expanded, bbox_expanded_mp, image_shape_2d, angle_mp, target_size,
                key_landmarks=mp_key_landmarks, show_all=True
            )
        else:
            face_mp_with_landmarks = None

        # Save face crops (with landmarks)
        face_if_path = output_dir / 'faces_insightface' / f"{image_path.stem}_face0.jpg"
        face_if_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(face_if_path), cv2.cvtColor(face_if_with_landmarks, cv2.COLOR_RGB2BGR))

        if not mp_failed and face_mp is not None:
            face_mp_path = output_dir / 'faces_mediapipe' / f"{image_path.stem}_face0.jpg"
            face_mp_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(face_mp_path), cv2.cvtColor(face_mp_with_landmarks, cv2.COLOR_RGB2BGR))
            face_mp_path_str = str(face_mp_path.relative_to(output_dir))
        else:
            face_mp_path_str = None

        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'rotation_mediapipe': round(angle_mp, 2) if not mp_failed else None,
            'rotation_insightface': round(angle_if, 2),
            'delta': round(delta, 2) if delta is not None else None,
            'mediapipe_failed': mp_failed,
            'face_mediapipe_path': face_mp_path_str,
            'face_insightface_path': str(face_if_path.relative_to(output_dir))
        }

    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return None


def generate_html_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    output_dir: Path
) -> None:
    """Generate HTML report with comparison table.

    Args:
        results: List of result dicts from process_image
        output_path: Path to save HTML file
        output_dir: Output directory for relative paths
    """
    if not results:
        logger.warning("No results to generate report")
        return

    # Compute summary statistics
    deltas = [r['delta'] for r in results if r['delta'] is not None]
    angles_mp = [r['rotation_mediapipe'] for r in results if r['rotation_mediapipe'] is not None]
    angles_if = [r['rotation_insightface'] for r in results]
    mp_failures = [r for r in results if r.get('mediapipe_failed', False)]

    mean_delta = np.mean(deltas) if deltas else 0.0
    max_delta = np.max(deltas) if deltas else 0.0
    mean_angle_mp = np.mean(angles_mp) if angles_mp else 0.0
    mean_angle_if = np.mean(angles_if) if angles_if else 0.0

    # Generate table rows
    rows_html = []
    for r in results:
        # Load face images from disk
        face_if_path = output_dir / r['face_insightface_path']
        face_if = cv2.imread(str(face_if_path))
        if face_if is not None:
            face_if = cv2.cvtColor(face_if, cv2.COLOR_BGR2RGB)
        face_if_b64 = image_to_base64(face_if) if face_if is not None else ''
        
        # MediaPipe face (may not exist if failed)
        mp_failed = r.get('mediapipe_failed', False)
        if mp_failed or r['face_mediapipe_path'] is None:
            face_mp_b64 = ''
            face_mp_path_display = 'MediaPipe failed'
            rotation_mp_display = '<span style="color: red; font-weight: bold;">FAILED</span>'
            delta_display = 'N/A'
            delta_class = 'mp-failed'
        else:
            face_mp_path = output_dir / r['face_mediapipe_path']
            face_mp = cv2.imread(str(face_mp_path))
            if face_mp is not None:
                face_mp = cv2.cvtColor(face_mp, cv2.COLOR_BGR2RGB)
            face_mp_b64 = image_to_base64(face_mp) if face_mp is not None else ''
            face_mp_path_display = r['face_mediapipe_path']
            rotation_mp_display = f"{r['rotation_mediapipe']:.2f}°"
            delta_display = f"{r['delta']:.2f}°" if r['delta'] is not None else 'N/A'
            delta_class = 'high-delta' if r['delta'] is not None and r['delta'] > 10 else ''

        row = f"""
        <tr class="{delta_class}">
            <td>{r['image_name']}</td>
            <td>{rotation_mp_display}</td>
            <td>{r['rotation_insightface']:.2f}°</td>
            <td>{delta_display}</td>
            <td>
                {f'<img src="{face_mp_b64}" alt="MediaPipe" style="max-width: 150px; max-height: 150px;" />' if face_mp_b64 else '<span style="color: red;">No image</span>'}
                <br/>
                <small>{face_mp_path_display}</small>
            </td>
            <td>
                <img src="{face_if_b64}" alt="InsightFace" style="max-width: 150px; max-height: 150px;" />
                <br/>
                <small>{r['face_insightface_path']}</small>
            </td>
        </tr>
        """
        rows_html.append(row)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Face Alignment Benchmark</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        tr.high-delta {{
            background-color: #fff3cd;
        }}
        tr.mp-failed {{
            background-color: #f8d7da;
        }}
        img {{
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        small {{
            color: #666;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Alignment Benchmark: MediaPipe vs InsightFace</h1>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{mean_delta:.2f}°</div>
                <div class="stat-label">Mean Delta</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{max_delta:.2f}°</div>
                <div class="stat-label">Max Delta</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{mean_angle_mp:.2f}° / {mean_angle_if:.2f}°</div>
                <div class="stat-label">Mean Angles (MP / IF)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(mp_failures)}</div>
                <div class="stat-label">MediaPipe Failures</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Image</th>
                    <th>Rotation MediaPipe</th>
                    <th>Rotation InsightFace</th>
                    <th>Delta</th>
                    <th>Face MediaPipe</th>
                    <th>Face InsightFace</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark MediaPipe vs InsightFace face alignment')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to image dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs/face_alignment_benchmark',
                        help='Output directory (default: outputs/face_alignment_benchmark)')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Crop margin around face (default: 0.2 = 20%%)')
    parser.add_argument('--target-size', type=int, default=256,
                        help='Resize crop to this size (default: 256)')
    parser.add_argument('--model-name', type=str, default='buffalo_l',
                        help='InsightFace model name (default: buffalo_l)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device for InsightFace (default: cpu)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    # Initialize face analyzer
    face_analyzer = InsightFaceFaceAnalyzer({
        'model_name': args.model_name,
        'device': args.device,
        'detection_threshold': 0.5
    })

    # Initialize MediaPipe FaceMesh once (reused across all images for performance)
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    logger.info("MediaPipe FaceMesh initialised")

    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in dataset_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {dataset_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {dataset_dir}")

    # Process images
    results = []
    skipped_no_face = 0
    skipped_no_landmarks = 0
    skipped_crop_failed = 0
    skipped_error = 0
    
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
        result = process_image(
            image_path,
            face_analyzer,
            output_dir,
            args.margin,
            args.target_size,
            face_mesh=face_mesh
        )
        if result:
            results.append(result)
        else:
            # Count skip reasons (approximate - actual reason is logged in process_image)
            skipped_error += 1

    logger.info("=" * 60)
    logger.info(f"Processing Summary:")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Successfully processed: {len(results)}")
    logger.info(f"  Skipped: {len(image_files) - len(results)}")
    logger.info("=" * 60)
    logger.info("Note: Check warnings above for details on skipped images")
    logger.info("  - 'No face detected' = InsightFace couldn't find a face")
    logger.info("  - 'MediaPipe failed' = MediaPipe couldn't detect landmarks")
    logger.info("  - 'Failed to crop' = Rotation/cropping failed")
    logger.info("=" * 60)

    # Save results JSON
    results_json = {
        'total_images': len(image_files),
        'processed_images': len(results),
        'results': results
    }
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Generate HTML report
    html_path = output_dir / 'comparison.html'
    generate_html_report(results, html_path, output_dir)
    logger.info(f"HTML report: {html_path}")


if __name__ == '__main__':
    main()
