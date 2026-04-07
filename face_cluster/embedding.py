"""InsightFace embedding extraction wrapper."""

import logging
from typing import List, Optional, Literal
import numpy as np
from pathlib import Path

from face_cluster.types import FaceRecord

logger = logging.getLogger(__name__)

FaceOrdering = Literal['detection_order', 'reading_order', 'area', 'confidence']


class InsightFaceEmbedder:
    """InsightFace wrapper for face detection and embedding extraction.

    Uses buffalo_l model by default for high-quality embeddings.
    """

    def __init__(
        self,
        model_name: str = 'buffalo_l',
        ctx_id: int = -1,
        face_ordering: FaceOrdering = 'detection_order'
    ):
        """Initialize InsightFace model.

        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            ctx_id: GPU device ID (-1 for CPU)
            face_ordering: Face ordering convention ('detection_order', 'reading_order', 'area', 'confidence')
        """
        try:
            import insightface
        except ImportError:
            raise ImportError("Install insightface: pip install insightface onnxruntime")

        self.model_name = model_name
        self.ctx_id = ctx_id
        self.face_ordering = face_ordering

        logger.info(f"Loading InsightFace model: {model_name}")
        self.app = insightface.app.FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info(f"InsightFace model loaded (face_ordering={face_ordering})")

    def detect_and_embed(
        self,
        image_paths: List[str],
        extract_pose: bool = True
    ) -> List[FaceRecord]:
        """Detect faces and extract embeddings from images.

        Args:
            image_paths: List of image file paths
            extract_pose: Whether to extract pose (yaw/pitch/roll) from landmarks

        Returns:
            List of FaceRecord objects with embeddings
        """
        import cv2
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener

        # Register HEIC/HEIF support
        register_heif_opener()

        all_faces = []
        face_id_counter = 0

        for image_path in image_paths:
            # Load image using PIL (supports HEIC)
            try:
                with Image.open(str(image_path)) as pil_img:
                    # Apply EXIF transpose
                    pil_img = ImageOps.exif_transpose(pil_img)

                    # Convert to RGB if needed
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')

                    # Convert PIL to numpy array (RGB)
                    img_rgb = np.array(pil_img)
            except Exception as e:
                logger.warning(f"Failed to load image: {image_path} - {e}")
                continue

            # Detect faces
            faces = self.app.get(img_rgb)

            # Apply face ordering convention
            faces = self._sort_faces(faces)

            for face_idx, face in enumerate(faces):
                # Extract bbox
                bbox = face.bbox.astype(float)
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                # Extract landmarks (5 points)
                landmarks = face.kps if hasattr(face, 'kps') else None

                # Extract embedding (already normalized by InsightFace)
                embedding = face.embedding
                embedding_normalized = embedding / np.linalg.norm(embedding)

                # Extract pose from InsightFace's 1k3d68 model (pitch, yaw, roll).
                # Reorder to (yaw, pitch, roll) to match FaceRecord convention.
                pose = None
                if extract_pose:
                    raw_pose = getattr(face, 'pose', None)
                    if raw_pose is not None and len(raw_pose) == 3:
                        pitch, yaw, roll = float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])
                        pose = (yaw, pitch, roll)

                # Create aligned face crop (112x112 as per ArcFace standard)
                aligned_face = None
                if landmarks is not None:
                    from insightface.utils import face_align
                    aligned_face = face_align.norm_crop(img_rgb, landmarks)

                face_record = FaceRecord(
                    face_id=face_id_counter,
                    image_id=Path(image_path).stem,
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    aligned_face=aligned_face,
                    embedding=embedding,
                    embedding_normalized=embedding_normalized,
                    pose=pose,
                    blur_score=0.0,  # Will be computed by QualityGater
                    area=area,
                    is_core=False,
                    image_path=str(image_path),
                    face_index=face_idx
                )

                all_faces.append(face_record)
                face_id_counter += 1

        logger.info(f"Detected {len(all_faces)} faces from {len(image_paths)} images")
        return all_faces

    def _sort_faces(self, faces: List) -> List:
        """Sort faces according to configured ordering convention.

        Args:
            faces: List of InsightFace face objects

        Returns:
            Sorted list of faces
        """
        if self.face_ordering == 'detection_order':
            return faces

        from sim_bench.utils.face_ordering import (
            sort_faces_reading_order,
            sort_faces_by_area,
            sort_faces_by_confidence
        )

        if self.face_ordering == 'reading_order':
            return sort_faces_reading_order(faces)
        elif self.face_ordering == 'area':
            return sort_faces_by_area(faces)
        elif self.face_ordering == 'confidence':
            return sort_faces_by_confidence(faces)
        else:
            logger.warning(f"Unknown face_ordering: {self.face_ordering}, using detection order")
            return faces

    def get_embedding(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from a single face image (already cropped/aligned).

        Args:
            img_rgb: Face image in RGB format (can be any size, will be processed)

        Returns:
            512-dim embedding vector, or None if no face detected
        """
        try:
            import cv2

            # Check image size first
            h, w = img_rgb.shape[:2]

            # For small images (≤224x224), assume pre-aligned crop and use recognition model directly
            # This ensures consistent embeddings (detection can be non-deterministic on crops)
            if h <= 224 and w <= 224:
                logger.info(f"[OK] Recognition model path: {w}x{h} crop, using direct embedding")

                # Resize to 112x112 (ArcFace standard input size)
                img_resized = cv2.resize(img_rgb, (112, 112))

                # Convert to BGR for InsightFace
                img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

                # Use recognition model directly
                embedding = self.app.models['recognition'].get_feat(img_bgr)
                # Ensure 1D array (flatten if needed)
                embedding = np.asarray(embedding).flatten()
                return embedding / np.linalg.norm(embedding)

            # For larger images, use detection to find face first
            logger.info(f"[OK] Detection path for {w}x{h} image")
            faces = self.app.get(img_rgb)
            if len(faces) > 0:
                embedding = faces[0].embedding
                # Ensure 1D array (flatten if needed)
                embedding = np.asarray(embedding).flatten()
                return embedding / np.linalg.norm(embedding)

            # No face detected
            return None

        except Exception as e:
            logger.debug(f"Failed to extract embedding: {e}")
            return None

    def embed_faces(self, faces: List[FaceRecord]) -> List[FaceRecord]:
        """Extract embeddings from pre-cropped face images.

        Args:
            faces: List of FaceRecord objects with aligned_face set

        Returns:
            Updated FaceRecord objects with embeddings
        """
        for face in faces:
            if face.aligned_face is None:
                logger.warning(f"No aligned face for face_id {face.face_id}")
                continue

            # Use get_embedding for consistency
            embedding = self.get_embedding(face.aligned_face)
            if embedding is None:
                logger.warning(f"No face detected in aligned crop for face_id {face.face_id}")
                continue

            face.embedding = embedding
            face.embedding_normalized = embedding

        return faces
