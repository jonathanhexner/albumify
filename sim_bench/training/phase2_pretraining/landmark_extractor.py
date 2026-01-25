"""
Extract and cache facial landmarks from AffectNet using MediaPipe.

Extracts 5-10 key landmarks (left eye, right eye, nose, mouth corners)
and saves them alongside images for efficient training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import cv2
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Key landmark indices from MediaPipe FaceMesh (468 points)
# Using well-known facial feature points
LANDMARK_INDICES = {
    'left_eye_center': 468,  # Left eye center (if refine_landmarks=True)
    'right_eye_center': 473,  # Right eye center (if refine_landmarks=True)
    'nose_tip': 4,  # Nose tip
    'mouth_left': 61,  # Left mouth corner
    'mouth_right': 291,  # Right mouth corner
}

# Alternative: 10-point set using standard FaceMesh indices
LANDMARK_INDICES_10 = {
    'left_eye_left': 33,
    'left_eye_right': 133,
    'right_eye_left': 362,
    'right_eye_right': 263,
    'nose_tip': 4,
    'nose_bottom': 2,
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
}


class LandmarkExtractor:
    """Extract facial landmarks using MediaPipe."""

    def __init__(self, num_landmarks: int = 5):
        """
        Initialize landmark extractor.
        
        Args:
            num_landmarks: Number of landmarks to extract (5 or 10)
        """
        self.num_landmarks = num_landmarks
        self.landmark_map = LANDMARK_INDICES if num_landmarks == 5 else LANDMARK_INDICES_10
        self._face_mesh = None
        logger.info(f"LandmarkExtractor initialized (num_landmarks={num_landmarks})")

    def _load_mediapipe(self):
        """Lazy load MediaPipe FaceMesh."""
        if self._face_mesh is not None:
            return
        
        import mediapipe as mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.debug("MediaPipe FaceMesh loaded")

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image with EXIF orientation correction."""
        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
        
        if img is None or img.size == 0:
            raise ValueError(f"Could not load image: {image_path}")
        return img

    def extract_landmarks(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract landmarks from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Landmarks as (num_landmarks, 2) array in normalized [0,1] coordinates,
            or None if no face detected
        """
        self._load_mediapipe()
        image = self._load_image(image_path)
        
        results = self._face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        
        extracted = []
        for name, idx in sorted(self.landmark_map.items()):
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = lm.x
                y = lm.y
                extracted.append([x, y])
            else:
                logger.warning(f"Landmark index {idx} out of range (max: {len(landmarks.landmark)-1})")
                return None
        
        if len(extracted) != self.num_landmarks:
            logger.warning(f"Expected {self.num_landmarks} landmarks, got {len(extracted)}")
            return None
        
        return np.array(extracted, dtype=np.float32)

    def extract_batch(self, image_paths: List[Path], cache_dir: Optional[Path] = None) -> Dict[Path, Optional[np.ndarray]]:
        """
        Extract landmarks for multiple images with optional caching.
        
        Args:
            image_paths: List of image paths
            cache_dir: Optional directory to cache landmarks
        
        Returns:
            Dict mapping image_path -> landmarks or None
        """
        results = {}
        cache_file = None
        
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "landmarks.json"
            
            cached = self._load_cache(cache_file)
        else:
            cached = {}
        
        for img_path in image_paths:
            img_path = Path(img_path)
            
            if img_path in cached:
                results[img_path] = cached[img_path]
                continue
            
            landmarks = self.extract_landmarks(img_path)
            results[img_path] = landmarks
            
            if cache_file and landmarks is not None:
                cached[img_path] = landmarks.tolist()
        
        if cache_file and cached:
            self._save_cache(cache_file, cached)
        
        return results

    def _load_cache(self, cache_file: Path) -> Dict[Path, Optional[np.ndarray]]:
        """Load cached landmarks."""
        if not cache_file.exists():
            return {}
        
        with open(cache_file) as f:
            data = json.load(f)
        
        cached = {}
        for path_str, landmarks_list in data.items():
            path = Path(path_str)
            if landmarks_list is not None:
                cached[path] = np.array(landmarks_list, dtype=np.float32)
            else:
                cached[path] = None
        
        logger.info(f"Loaded {len(cached)} cached landmarks from {cache_file}")
        return cached

    def _save_cache(self, cache_file: Path, cache_data: Dict):
        """Save landmarks to cache."""
        serializable = {}
        for path, landmarks in cache_data.items():
            if landmarks is not None:
                serializable[str(path)] = landmarks.tolist()
            else:
                serializable[str(path)] = None
        
        with open(cache_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Saved {len(serializable)} landmarks to {cache_file}")
