"""
Complete photo analysis pipeline with specialized models.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import numpy as np

from sim_bench.photo_analysis.clip_tagger import CLIPTagger
from sim_bench.specialized_models import create_specialized_model

logger = logging.getLogger(__name__)


class PhotoAnalysisPipeline:
    """
    Complete photo analysis pipeline.
    
    Combines CLIP tagging, routing, and specialized models (faces, landmarks).
    """
    
    def __init__(
        self,
        clip_config: Optional[Dict[str, Any]] = None,
        face_config: Optional[Dict[str, Any]] = None,
        landmark_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            clip_config: Configuration for CLIP tagger
            face_config: Configuration for face model
            landmark_config: Configuration for landmark model
        """
        clip_config = clip_config or {}
        self.clip_tagger = CLIPTagger(**clip_config)
        
        # Initialize specialized models (lazy loading)
        self.face_config = face_config or {}
        self.landmark_config = landmark_config or {}
        self._face_model = None
        self._landmark_model = None
    
    def _get_face_model(self):
        """Lazy load face model."""
        if self._face_model is None:
            self._face_model = create_specialized_model('face', **self.face_config)
        return self._face_model
    
    def _get_landmark_model(self):
        """Lazy load landmark model."""
        if self._landmark_model is None:
            self._landmark_model = create_specialized_model('landmark', **self.landmark_config)
        return self._landmark_model
    
    def analyze_with_specialized(
        self,
        image_paths: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze images with CLIP and specialized models.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dict mapping image_path -> complete analysis including:
                - clip_analysis: CLIP tagging results
                - face_results: Face detection/embeddings (if applicable)
                - landmark_results: Landmark embeddings (if applicable)
        """
        # Step 1: CLIP analysis
        logger.info(f"Step 1/3: Analyzing {len(image_paths)} images with CLIP...")
        clip_results = self.clip_tagger.analyze_batch(
            image_paths,
            batch_size=8,
            verbose=False
        )
        
        # Step 2: Collect images that need specialized models
        images_needing_faces = []
        images_needing_landmarks = []
        
        for img_path, analysis in clip_results.items():
            routing = analysis.get('routing', {})
            if routing.get('needs_face_detection', False):
                images_needing_faces.append(img_path)
            if routing.get('needs_landmark_detection', False):
                images_needing_landmarks.append(img_path)
        
        # Step 3: Run specialized models
        face_results = {}
        landmark_results = {}
        
        if images_needing_faces:
            logger.info(f"Step 2/3: Processing {len(images_needing_faces)} images with face model...")
            face_model = self._get_face_model()
            face_results = face_model.process_batch(images_needing_faces)
        
        if images_needing_landmarks:
            logger.info(f"Step 3/3: Processing {len(images_needing_landmarks)} images with landmark model...")
            landmark_model = self._get_landmark_model()
            landmark_results = landmark_model.process_batch(images_needing_landmarks)
        
        # Step 4: Combine results
        combined_results = {}
        for img_path in image_paths:
            combined_results[img_path] = {
                'clip_analysis': clip_results.get(img_path, {}),
                'face_results': face_results.get(img_path),
                'landmark_results': landmark_results.get(img_path)
            }
        
        logger.info("Analysis complete")
        return combined_results
    
    def extract_embeddings_for_clustering(
        self,
        analysis_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract all embeddings for clustering.
        
        Args:
            analysis_results: Results from analyze_with_specialized()
            
        Returns:
            Dict with:
                - scene_embeddings: CLIP scene embeddings
                - face_embeddings: Face embeddings (if available)
                - landmark_embeddings: Landmark embeddings (if available)
        """
        # Get CLIP model for scene embeddings
        clip_model = self.clip_tagger.clip
        
        scene_embeddings = {}
        face_embeddings = {}
        landmark_embeddings = {}
        
        # Batch encode all images for scene embeddings
        image_paths = list(analysis_results.keys())
        scene_embs = clip_model.encode_images(
            [results.get('clip_analysis', {}).get('path', path) for path in image_paths],
            batch_size=32
        )
        
        for img_path, scene_emb in zip(image_paths, scene_embs):
            scene_embeddings[img_path] = scene_emb
            
            # Face embeddings
            results = analysis_results[img_path]
            face_res = results.get('face_results')
            if face_res and face_res.get('embeddings'):
                face_embeddings[img_path] = np.array(face_res['embeddings'])
            
            # Landmark embeddings
            landmark_res = results.get('landmark_results')
            if landmark_res and 'embedding' in landmark_res:
                landmark_embeddings[img_path] = landmark_res['embedding']
        
        return {
            'scene_embeddings': scene_embeddings,
            'face_embeddings': face_embeddings,
            'landmark_embeddings': landmark_embeddings
        }

