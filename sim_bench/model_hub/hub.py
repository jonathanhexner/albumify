"""
Unified interface to all image analysis models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from sim_bench.model_hub.types import ImageMetrics

logger = logging.getLogger(__name__)


class ModelHub:
    """
    Unified interface to all image analysis models.

    Config-only constructor - receives full config dict.
    Provides single entry point for quality assessment, portrait analysis,
    feature extraction, and clustering.

    Example:
        config = get_global_config().to_dict()
        hub = ModelHub(config)

        # Single image analysis
        metrics = hub.analyze_image(Path("photo.jpg"))

        # Batch analysis
        all_metrics = hub.analyze_batch([Path("img1.jpg"), Path("img2.jpg")])
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelHub with configuration.

        Args:
            config: Full configuration dictionary
        """
        self._config = config
        self._device = config.get('device', 'cpu')

        # Lazy-loaded model instances
        self._iqa_model = None
        self._ava_model = None
        self._siamese_model = None
        self._portrait_analyzer = None
        self._feature_extractor = None
        self._clustering_method = None

        logger.info(f"ModelHub initialized (device={self._device})")

    # =========================================================================
    # Model Accessors (Lazy Loading)
    # =========================================================================

    def _get_iqa_model(self):
        """Lazy load IQA model."""
        if self._iqa_model is None:
            from sim_bench.quality_assessment.rule_based import RuleBasedQuality
            self._iqa_model = RuleBasedQuality()
            logger.info("Loaded IQA model (RuleBasedQuality)")
        return self._iqa_model

    def _get_portrait_analyzer(self):
        """Lazy load portrait analyzer."""
        if self._portrait_analyzer is None:
            from sim_bench.portrait_analysis import MediaPipePortraitAnalyzer
            self._portrait_analyzer = MediaPipePortraitAnalyzer(self._config)
            logger.info("Loaded portrait analyzer (MediaPipe)")
        return self._portrait_analyzer

    def _get_feature_extractor(self, method: str = None):
        """Lazy load feature extractor."""
        method = method or self._config.get('album', {}).get('clustering', {}).get('feature_method', 'dinov2')

        if self._feature_extractor is None:
            from sim_bench.feature_extraction.base import load_method
            extractor_config = {
                'method': method,
                'device': self._device,
                'batch_size': self._config.get('performance', {}).get('max_batch_size', 32)
            }
            self._feature_extractor = load_method(method, extractor_config)
            logger.info(f"Loaded feature extractor ({method})")
        return self._feature_extractor

    def _get_clustering_method(self):
        """Lazy load clustering method."""
        if self._clustering_method is None:
            from sim_bench.clustering.base import load_clustering_method
            clustering_config = self._config.get('album', {}).get('clustering', {})
            full_config = {
                'algorithm': clustering_config.get('method', 'hdbscan'),
                'params': {
                    'min_cluster_size': clustering_config.get('min_cluster_size', 5)
                },
                'output': {}
            }
            self._clustering_method = load_clustering_method(full_config)
            logger.info(f"Loaded clustering method ({full_config['algorithm']})")
        return self._clustering_method

    # =========================================================================
    # Quality Assessment
    # =========================================================================

    def score_quality(self, image_path: Path) -> Dict[str, float]:
        """
        Get technical quality scores using IQA.

        Args:
            image_path: Path to image

        Returns:
            Dict with 'overall', 'sharpness', 'exposure', etc.
        """
        model = self._get_iqa_model()
        detailed = model.get_detailed_scores(str(image_path))
        return {
            'overall': detailed['overall'],
            'sharpness': detailed['sharpness_normalized'],
            'exposure': detailed['exposure'],
            'colorfulness': detailed['colorfulness_normalized'],
            'contrast': detailed['contrast']
        }

    def score_aesthetics(self, image_path: Path) -> Optional[float]:
        """
        Get aesthetic score using AVA model (if available).

        Args:
            image_path: Path to image

        Returns:
            Aesthetic score (1-10) or None if model not configured
        """
        ava_checkpoint = self._config.get('quality_assessment', {}).get('ava_checkpoint')
        if not ava_checkpoint:
            return None

        if self._ava_model is None:
            from sim_bench.image_quality_models.ava_model_wrapper import AVAQualityModel
            self._ava_model = AVAQualityModel(Path(ava_checkpoint), self._device)
            logger.info("Loaded AVA model")

        return self._ava_model.score_image(image_path)

    def compare_images(self, img1: Path, img2: Path) -> Dict[str, Any]:
        """
        Compare two images using trained Siamese model (if configured).

        Returns:
            Dict with 'winner' (1 or 2), 'confidence', 'scores'
        """
        siamese_checkpoint = self._config.get('quality_assessment', {}).get('siamese_checkpoint')
        
        if siamese_checkpoint:
            if self._siamese_model is None:
                from sim_bench.image_quality_models.siamese_model_wrapper import SiameseQualityModel
                self._siamese_model = SiameseQualityModel(Path(siamese_checkpoint), self._device)
                logger.info("Loaded Siamese comparison model")
            
            result = self._siamese_model.compare_images(img1, img2)
            return {
                'winner': 1 if result['prediction'] == 1 else 2,
                'confidence': result['confidence'],
                'scores': [result.get('score_img1', 0.5), result.get('score_img2', 0.5)]
            }
        
        score1 = self.score_quality(img1)['overall']
        score2 = self.score_quality(img2)['overall']

        winner = 1 if score1 > score2 else 2
        confidence = abs(score1 - score2) / max(score1, score2, 0.001)

        return {
            'winner': winner,
            'confidence': min(confidence, 1.0),
            'scores': [score1, score2]
        }

    # =========================================================================
    # Portrait Analysis
    # =========================================================================

    def analyze_portrait(self, image_path: Path):
        """
        Analyze portrait metrics (face, eyes, smile).

        Args:
            image_path: Path to image

        Returns:
            PortraitMetrics from portrait_analysis module
        """
        analyzer = self._get_portrait_analyzer()
        return analyzer.analyze_image(image_path)

    # =========================================================================
    # Feature Extraction & Clustering
    # =========================================================================

    def extract_features(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract features for clustering.

        Args:
            image_paths: List of image paths

        Returns:
            Feature matrix [n_images, feature_dim]
        """
        extractor = self._get_feature_extractor()
        str_paths = [str(p) for p in image_paths]
        return extractor.extract_features(str_paths)

    def cluster_images(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster images based on features.

        Args:
            features: Feature matrix [n_images, feature_dim]

        Returns:
            (labels, stats) tuple
        """
        clusterer = self._get_clustering_method()
        return clusterer.cluster(features)

    # =========================================================================
    # Unified Analysis
    # =========================================================================

    def analyze_image(
        self,
        image_path: Path,
        include_quality: bool = True,
        include_portrait: bool = True
    ) -> ImageMetrics:
        """
        Complete analysis of single image.

        Args:
            image_path: Path to image
            include_quality: Include IQA scores
            include_portrait: Include portrait analysis

        Returns:
            ImageMetrics with all available scores
        """
        image_path = Path(image_path)
        metrics = ImageMetrics(image_path=str(image_path))

        if include_quality:
            quality = self.score_quality(image_path)
            metrics.iqa_score = quality['overall']
            metrics.sharpness = quality['sharpness']
            metrics.exposure = quality['exposure']
            metrics.colorfulness = quality['colorfulness']
            metrics.contrast = quality['contrast']

            ava = self.score_aesthetics(image_path)
            if ava is not None:
                metrics.ava_score = ava

        if include_portrait:
            portrait = self.analyze_portrait(image_path)
            metrics.has_face = portrait.has_face
            metrics.num_faces = portrait.num_faces
            metrics.is_portrait = portrait.is_portrait

            if portrait.eye_state:
                metrics.eyes_open = portrait.eye_state.both_eyes_open
                metrics.eye_aspect_ratio = (
                    portrait.eye_state.left_ear + portrait.eye_state.right_ear
                ) / 2

            if portrait.smile_state:
                metrics.is_smiling = portrait.smile_state.is_smiling
                metrics.smile_score = portrait.smile_state.smile_score

        return metrics

    def analyze_batch(
        self,
        image_paths: List[Path],
        thumbnails: Optional[Dict[Path, Dict[str, Path]]] = None,
        include_quality: bool = True,
        include_portrait: bool = True,
        include_features: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, ImageMetrics]:
        """
        Analyze batch of images.

        Args:
            image_paths: List of image paths
            thumbnails: Optional thumbnail mapping for efficient processing
            include_quality: Include IQA scores
            include_portrait: Include portrait analysis
            include_features: Include scene embeddings
            progress_callback: Optional callback(operation, current, total, image_name)
                             New signature for detailed progress reporting

        Returns:
            Dict mapping image_path -> ImageMetrics
        """
        results = {}
        total = len(image_paths)

        for idx, path in enumerate(image_paths):
            image_path = Path(path)
            
            # Get appropriate image paths for each operation
            quality_img = image_path
            portrait_img = image_path
            
            if thumbnails and image_path in thumbnails:
                quality_img = thumbnails[image_path].get('quality', image_path)
                portrait_img = thumbnails[image_path].get('portrait', image_path)
            
            # Initialize metrics
            image_path_str = str(image_path)
            metrics = ImageMetrics(image_path=image_path_str)
            
            # Quality assessment
            if include_quality:
                if progress_callback:
                    progress_callback("IQA Quality", idx, total, image_path.name)
                quality = self.score_quality(quality_img)
                metrics.iqa_score = quality['overall']
                metrics.sharpness = quality['sharpness']
                metrics.exposure = quality['exposure']
                metrics.colorfulness = quality['colorfulness']
                metrics.contrast = quality['contrast']
                
                # AVA aesthetics
                if progress_callback:
                    progress_callback("AVA Aesthetics", idx, total, image_path.name)
                ava = self.score_aesthetics(quality_img)
                if ava is not None:
                    metrics.ava_score = ava
            
            # Portrait analysis
            if include_portrait:
                if progress_callback:
                    progress_callback("Portrait Detection", idx, total, image_path.name)
                portrait = self.analyze_portrait(portrait_img)
                metrics.has_face = portrait.has_face
                metrics.num_faces = portrait.num_faces
                metrics.is_portrait = portrait.is_portrait
                
                if portrait.eye_state:
                    metrics.eyes_open = portrait.eye_state.both_eyes_open
                    metrics.eye_aspect_ratio = (
                        portrait.eye_state.left_ear + portrait.eye_state.right_ear
                    ) / 2
                
                if portrait.smile_state:
                    metrics.is_smiling = portrait.smile_state.is_smiling
                    metrics.smile_score = portrait.smile_state.smile_score
            
            results[image_path_str] = metrics

        # Feature extraction (batch operation)
        if include_features:
            if progress_callback:
                progress_callback("Feature Extraction", total, total, "batch")
            
            # Use thumbnails for features if available
            feature_paths = image_paths
            if thumbnails:
                feature_paths = [
                    thumbnails[Path(p)].get('features', Path(p)) 
                    if Path(p) in thumbnails else Path(p)
                    for p in image_paths
                ]
            
            features = self.extract_features(feature_paths)
            for idx, path in enumerate(image_paths):
                results[str(path)].scene_embedding = features[idx]

        portraits = sum(1 for m in results.values() if m.is_portrait)
        logger.info(f"Analyzed {total} images: {portraits} portraits, quality={include_quality}")

        return results
