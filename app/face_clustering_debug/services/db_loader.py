"""DBLoader - Load clustering data from sim_bench SQLite database."""

import io
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from app.face_clustering_debug.models.schemas import (
    FaceInfo,
    ClusterInfo,
    MergeDecision,
    AttachDecision,
    ClusteringResult,
)

logger = logging.getLogger(__name__)


class DBLoader:
    """Load clustering data from sim_bench SQLite database."""

    def __init__(self, album_id: str, pipeline_run_id: Optional[str] = None):
        self.album_id = album_id
        self.pipeline_run_id = pipeline_run_id
        self._db_path = Path.home() / ".sim_bench" / "sim_bench.db"
        self._faces_cache: Optional[List[FaceInfo]] = None
        self._embeddings_cache: Optional[np.ndarray] = None
        # Raw detection data keyed by face index for on-the-fly cropping
        self._face_raw: Dict[int, Dict[str, Any]] = {}

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")
        return sqlite3.connect(self._db_path)

    def load_embeddings(self) -> np.ndarray:
        """Load face embeddings from universal_cache."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all face embeddings for this album
        # First, get album path
        cursor.execute("SELECT source_path FROM albums WHERE id = ?", (self.album_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Album not found: {self.album_id}")

        album_path = row[0].replace("\\", "/")  # Normalize to forward slashes

        # Get embeddings where image_path starts with album_path
        cursor.execute("""
            SELECT image_path, data_blob
            FROM universal_cache
            WHERE feature_type = 'face_embedding'
            AND image_path LIKE ?
            ORDER BY image_path
        """, (f"{album_path}%",))

        embeddings = []
        for row in cursor.fetchall():
            emb = np.frombuffer(row[1], dtype=np.float32)
            # Handle potential extra data (take first 512 dims)
            if len(emb) > 512:
                emb = emb[:512]
            embeddings.append(emb)

        conn.close()

        if not embeddings:
            raise ValueError(f"No embeddings found for album: {self.album_id}")

        self._embeddings_cache = np.array(embeddings)
        logger.info(f"Loaded {len(embeddings)} embeddings from database")
        return self._embeddings_cache

    def load_faces(self) -> List[FaceInfo]:
        """Load face metadata from universal_cache."""
        if self._faces_cache is not None:
            return self._faces_cache

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get album path
        cursor.execute("SELECT source_path FROM albums WHERE id = ?", (self.album_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Album not found: {self.album_id}")

        album_path = row[0].replace("\\", "/")  # Normalize to forward slashes

        # Get face detection data
        cursor.execute("""
            SELECT image_path, data_blob
            FROM universal_cache
            WHERE feature_type = 'insightface_detection'
            AND image_path LIKE ?
            ORDER BY image_path
        """, (f"{album_path}%",))

        faces = []
        face_index = 0

        for row in cursor.fetchall():
            image_path = row[0]
            detection_data = json.loads(row[1])

            for face_data in detection_data.get("faces", []):
                bbox_data = face_data.get("bbox", {})
                bbox = (
                    bbox_data.get("x", 0),
                    bbox_data.get("y", 0),
                    bbox_data.get("w", 0),
                    bbox_data.get("h", 0),
                )

                # Get pixel bbox for landmark normalization
                bbox_px_w = bbox_data.get("w_px", 1)
                bbox_px_h = bbox_data.get("h_px", 1)
                bbox_px_x = bbox_data.get("x_px", 0)
                bbox_px_y = bbox_data.get("y_px", 0)

                # Normalize landmarks to 0-1 range relative to face bbox
                landmarks = face_data.get("landmarks")
                if landmarks and isinstance(landmarks, list) and bbox_px_w > 0 and bbox_px_h > 0:
                    normalized = []
                    for p in landmarks[:5]:
                        x_norm = (p[0] - bbox_px_x) / bbox_px_w
                        y_norm = (p[1] - bbox_px_y) / bbox_px_h
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                        normalized.append((x_norm, y_norm))
                    landmarks = normalized
                elif landmarks:
                    landmarks = [(p[0], p[1]) for p in landmarks[:5]]

                self._face_raw[face_index] = {
                    "image_path": image_path,
                    "bbox": bbox_data,
                    "roll_angle": face_data.get("roll_angle", 0.0),
                    "landmarks": face_data.get("landmarks"),  # Keep original for 5-point alignment
                }
                faces.append(FaceInfo(
                    index=face_index,
                    image_path=image_path,
                    bbox=bbox,
                    confidence=face_data.get("confidence", 0.0),
                    crop_path=None,
                    landmarks=landmarks,
                    pose_angles=None,
                ))
                face_index += 1

        conn.close()
        self._faces_cache = faces
        logger.info(f"Loaded {len(faces)} faces from database")
        return self._faces_cache

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        """Load clustering result from people table.

        Note: DB stores final clustering results, not debug decisions.
        Merge/attach decisions are not available from DB.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get people for this album
        cursor.execute("""
            SELECT person_index, face_instances, face_count
            FROM people
            WHERE album_id = ?
            ORDER BY person_index
        """, (self.album_id,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        faces = self.load_faces()
        embeddings = self.load_embeddings()

        # Build labels array from people data
        labels = np.full(len(faces), -1)  # Default to noise
        clusters = []

        for person_index, face_instances_json, face_count in rows:
            face_instances = json.loads(face_instances_json) if face_instances_json else []
            face_indices = []

            # Map face instances to indices
            for instance in face_instances:
                img_path = instance.get("image_path", "")
                face_idx_in_img = instance.get("face_index", 0)

                # Find matching face in our faces list
                for i, face in enumerate(faces):
                    if face.image_path == img_path:
                        # Simple heuristic: use face index offset
                        face_indices.append(i)
                        labels[i] = person_index
                        break

            clusters.append(ClusterInfo(
                cluster_id=person_index,
                face_indices=face_indices,
                threshold=0.0,  # Not available from DB
                exemplar_indices=[],
            ))

        return ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            faces=faces,
            clusters=clusters,
            merge_decisions=[],  # Not available from DB
            attach_decisions=[],  # Not available from DB
            algorithm=method,
            params={},
            n_clusters=len(clusters),
            n_noise=int(np.sum(labels == -1)),
        )

    def get_available_methods(self) -> List[str]:
        """List available methods - DB only has final results."""
        return ["database"]  # DB results don't have method distinction

    def get_run_info(self) -> dict:
        """Return basic info — DB doesn't have benchmark run metadata."""
        return {
            "album_name": f"Album {self.album_id}",
            "album_path": "",
            "total_faces": 0,
            "timestamp": "",
        }

    def has_debug_data(self, method: str) -> bool:
        """DB loader never has detailed debug data."""
        return False

    def get_face_crop(self, face_index: int) -> Optional[bytes]:
        """Crop face on-the-fly with orientation detection and 5-point alignment.

        Uses the two-stage alignment approach:
        1. Detect face orientation (0°, 90°, 180°, 270°)
        2. Generous crop with bbox-based margin
        3. Rotate to upright
        4. 5-point affine alignment
        """
        from sim_bench.pipeline.steps.detect_face_orientation import detect_face_orientation
        from sim_bench.pipeline.steps.align_faces import align_face_with_orientation
        from sim_bench.pipeline.utils.image_cache import get_image_cache

        # Ensure face metadata is loaded
        if not self._face_raw:
            self.load_faces()

        raw = self._face_raw.get(face_index)
        if raw is None:
            return None

        image_path = raw["image_path"]
        bbox_data = raw["bbox"]
        landmarks = raw.get("landmarks")  # Original pixel coordinates

        cache = get_image_cache()
        img = cache.get(image_path)
        if img is None:
            return None

        crop = None

        # Use orientation-aware alignment if landmarks available
        if landmarks and len(landmarks) >= 5:
            # Detect orientation from landmarks
            orientation = detect_face_orientation(landmarks)

            # Build bbox dict for proper margin calculation
            bbox = {
                'x': int(bbox_data.get("x_px", 0)),
                'y': int(bbox_data.get("y_px", 0)),
                'w': int(bbox_data.get("w_px", 0)),
                'h': int(bbox_data.get("h_px", 0)),
            }

            # Use two-stage alignment with orientation correction
            crop = align_face_with_orientation(
                img, landmarks, orientation,
                target_size=256, initial_margin=0.5, bbox=bbox
            )

        # Fallback to simple bbox crop if alignment fails
        if crop is None:
            from sim_bench.pipeline.utils.face_alignment import align_and_crop_face
            bbox = {
                "x_px": int(bbox_data.get("x_px", 0)),
                "y_px": int(bbox_data.get("y_px", 0)),
                "w_px": int(bbox_data.get("w_px", 0)),
                "h_px": int(bbox_data.get("h_px", 0)),
            }
            roll_angle = raw.get("roll_angle", 0.0)
            crop = align_and_crop_face(img, bbox, roll_angle, margin=0.2, target_size=256)

        if crop is None:
            return None

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    def get_face_crop_raw(self, face_index: int) -> Optional[bytes]:
        """Get raw face crop (bbox only, no alignment) as JPEG bytes."""
        from sim_bench.pipeline.utils.image_cache import get_image_cache

        if not self._face_raw:
            self.load_faces()

        raw = self._face_raw.get(face_index)
        if raw is None:
            return None

        image_path = raw["image_path"]
        bbox_data = raw["bbox"]

        x = int(bbox_data.get("x_px", 0))
        y = int(bbox_data.get("y_px", 0))
        w = int(bbox_data.get("w_px", 0))
        h = int(bbox_data.get("h_px", 0))

        if w <= 0 or h <= 0:
            return None

        cache = get_image_cache()
        img = cache.get(image_path)
        if img is None:
            return None

        # Add margin (20%)
        margin = 0.2
        mw, mh = int(w * margin), int(h * margin)
        img_h, img_w = img.shape[:2]

        x1 = max(0, x - mw)
        y1 = max(0, y - mh)
        x2 = min(img_w, x + w + mw)
        y2 = min(img_h, y + h + mh)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to 256x256
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    def get_original_image_with_bbox(self, face_index: int, max_size: int = 800) -> Optional[bytes]:
        """Get original image with face bbox and landmarks drawn, resized for display."""
        from sim_bench.pipeline.utils.image_cache import get_image_cache

        if not self._face_raw:
            self.load_faces()

        raw = self._face_raw.get(face_index)
        if raw is None:
            return None

        image_path = raw["image_path"]
        bbox_data = raw["bbox"]
        landmarks = raw.get("landmarks")

        cache = get_image_cache()
        img = cache.get(image_path)
        if img is None:
            return None

        # Convert to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw bbox
        x = int(bbox_data.get("x_px", 0))
        y = int(bbox_data.get("y_px", 0))
        w = int(bbox_data.get("w_px", 0))
        h = int(bbox_data.get("h_px", 0))
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw landmarks
        if landmarks and len(landmarks) == 5:
            colors = [(0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)]
            for (lx, ly), color in zip(landmarks, colors):
                cv2.circle(img_bgr, (int(lx), int(ly)), 5, color, -1)
                cv2.circle(img_bgr, (int(lx), int(ly)), 7, (255, 255, 255), 2)

        # Resize for display
        img_h, img_w = img_bgr.shape[:2]
        scale = min(max_size / img_w, max_size / img_h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
