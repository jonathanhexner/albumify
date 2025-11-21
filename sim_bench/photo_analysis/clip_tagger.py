"""
CLIP-based photo tagging with 55 zero-shot prompts.

Analyzes images using vision-language similarity to extract:
- Scene type (outdoor, indoor, landmark, etc.)
- Quality indicators (sharp, blurry, well-exposed, etc.)
- Composition features (balanced, rule of thirds, etc.)
- Routing decisions (which specialized models to apply)
"""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import logging
import yaml
import numpy as np

from sim_bench.photo_analysis.base import PhotoAnalyzer
from sim_bench.vision_language import CLIPModel
from sim_bench.config import get_global_config


logger = logging.getLogger(__name__)


class CLIPTagger(PhotoAnalyzer):
    """
    CLIP-based zero-shot photo tagger.

    Uses 55 carefully designed prompts to extract comprehensive metadata
    from photos without any training. Routes images to specialized models
    based on content analysis.

    Example:
        >>> from sim_bench.photo_analysis import CLIPTagger
        >>>
        >>> tagger = CLIPTagger()
        >>> analysis = tagger.analyze_image("photo.jpg")
        >>> print(analysis['primary_tags'])  # ['outdoor', 'landscape', 'well_composed']
        >>> print(analysis['importance_score'])  # 0.82
        >>> print(analysis['routing']['needs_face_detection'])  # False
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        device: Optional[str] = None,
        enable_cache: bool = True,
        prompts_config: Optional[Union[str, Path]] = None
    ):
        """
        Initialize CLIP tagger.

        Args:
            model_name: CLIP model name (default from global config)
            pretrained: Pretrained checkpoint (default from global config)
            device: Device for computation (default from global config)
            enable_cache: Whether to cache results
            prompts_config: Path to prompts config YAML (default: configs/photo_analysis_prompts.yaml)
        """
        # Load global config
        config = get_global_config()

        # Set defaults from config
        if model_name is None:
            model_name = config.get('clip.model_name', 'ViT-B-32')
        if pretrained is None:
            pretrained = config.get('clip.pretrained', 'laion2b_s34b_b79k')
        if device is None:
            device = config.get('device', 'cpu')

        super().__init__(enable_cache=enable_cache, device=device)

        # Load CLIP model
        try:
            self.clip = CLIPModel(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                enable_cache=enable_cache
            )
            logger.info(f"Loaded CLIP model: {model_name} ({pretrained})")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

        # Load prompts configuration
        self._load_prompts_config(prompts_config)

        # Encode all prompts once (cached)
        self._encode_prompts()

        logger.info(
            f"CLIPTagger initialized: {self.total_prompts} prompts, "
            f"device={device}"
        )

    def _load_prompts_config(self, prompts_config: Optional[Union[str, Path]]) -> None:
        """Load prompts from YAML configuration."""
        if prompts_config is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent
            prompts_config = project_root / 'configs' / 'photo_analysis_prompts.yaml'
        else:
            prompts_config = Path(prompts_config)

        if not prompts_config.exists():
            raise FileNotFoundError(f"Prompts config not found: {prompts_config}")

        with open(prompts_config, 'r') as f:
            config = yaml.safe_load(f)

        # Extract prompts by category
        self.prompts = {
            'scene_content': config.get('scene_content', []),
            'quality_technical': config.get('quality_technical', []),
            'composition_aesthetic': config.get('composition_aesthetic', []),
            'human_focused': config.get('human_focused', [])
        }

        # Flatten all prompts
        self.all_prompts = []
        self.prompt_to_category = {}

        for category, prompts in self.prompts.items():
            for prompt in prompts:
                self.all_prompts.append(prompt)
                self.prompt_to_category[prompt] = category

        self.total_prompts = len(self.all_prompts)

        # Load thresholds and weights
        self.routing_thresholds = config.get('routing_thresholds', {})
        self.importance_weights = config.get('importance_weights', {
            'quality': 0.4,
            'composition': 0.3,
            'uniqueness': 0.3
        })

        logger.info(f"Loaded {self.total_prompts} prompts across {len(self.prompts)} categories")

    def _encode_prompts(self) -> None:
        """Encode all prompts to embeddings (cached)."""
        try:
            self.prompt_embeddings = self.clip.encode_texts(self.all_prompts)
            logger.debug(f"Encoded {len(self.all_prompts)} text prompts")
        except Exception as e:
            raise RuntimeError(f"Failed to encode prompts: {e}")

    def analyze_image(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze single image using CLIP zero-shot prompts.

        Args:
            image_path: Path to image file

        Returns:
            Dict with:
                - path: Image path
                - tags: Dict mapping all prompts to similarity scores
                - primary_tags: Top-5 tags
                - category_scores: Aggregated scores per category
                - importance_score: Overall importance (0-1)
                - routing: Which specialized models to trigger
                - metadata: Additional information
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image
        try:
            image_embedding = self.clip.encode_images([str(image_path)])[0]
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return self._get_empty_analysis(image_path)

        # Compute similarities to all prompts
        similarities = self.clip.compute_similarity(
            image_embedding.reshape(1, -1),
            self.prompt_embeddings
        )[0]

        # Create tags dict
        tags = {
            prompt: float(score)
            for prompt, score in zip(self.all_prompts, similarities)
        }

        # Get top-k tags
        top_indices = np.argsort(similarities)[::-1][:5]
        primary_tags = [self.all_prompts[i] for i in top_indices]

        # Aggregate scores by category
        category_scores = self._aggregate_by_category(tags)

        # Compute importance score
        importance_score = self._compute_importance_score(category_scores, tags)

        # Determine routing
        routing = self._determine_routing(tags, category_scores)

        # Build result
        return {
            'path': str(image_path),
            'tags': tags,
            'primary_tags': primary_tags,
            'category_scores': category_scores,
            'importance_score': importance_score,
            'routing': routing,
            'metadata': {
                'total_prompts': self.total_prompts,
                'top_5_scores': [float(similarities[i]) for i in top_indices]
            }
        }

    def _aggregate_by_category(self, tags: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate tag scores by category.

        Args:
            tags: Dict of prompt -> score

        Returns:
            Dict of category -> aggregated score
        """
        category_scores = {}

        for category, prompts in self.prompts.items():
            scores = [tags.get(prompt, 0.0) for prompt in prompts]
            # Use mean aggregation (could also use max or weighted)
            category_scores[category] = float(np.mean(scores))

        return category_scores

    def _compute_importance_score(
        self,
        category_scores: Dict[str, float],
        tags: Dict[str, float]
    ) -> float:
        """
        Compute overall importance score for photo.

        Higher score = more important for album (quality + composition + uniqueness)

        Args:
            category_scores: Aggregated scores by category
            tags: Individual tag scores

        Returns:
            Importance score (0-1)
        """
        # Quality component (high quality, sharp, well-exposed, good lighting)
        quality_prompts = [
            "a high-quality photo",
            "a sharp photo",
            "a well-exposed photo",
            "a photo with good lighting"
        ]
        quality_score = np.mean([tags.get(p, 0.0) for p in quality_prompts])

        # Composition component
        composition_prompts = [
            "a well-composed photograph",
            "a balanced composition",
            "a photo with good framing",
            "a rule of thirds composition"
        ]
        composition_score = np.mean([tags.get(p, 0.0) for p in composition_prompts])

        # Uniqueness component (inverse of common/boring content)
        # High scores on distinctive scenes = more unique
        uniqueness_score = category_scores.get('scene_content', 0.5)

        # Weighted combination
        importance = (
            self.importance_weights['quality'] * quality_score +
            self.importance_weights['composition'] * composition_score +
            self.importance_weights['uniqueness'] * uniqueness_score
        )

        return float(np.clip(importance, 0, 1))

    def _determine_routing(
        self,
        tags: Dict[str, float],
        category_scores: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Determine which specialized models to apply.

        Args:
            tags: Individual tag scores
            category_scores: Category aggregate scores

        Returns:
            Dict of model_name -> should_apply (bool)
        """
        routing = {}

        # Face detection: triggered by human-focused prompts
        face_threshold = self.routing_thresholds.get('face_detection', 0.6)
        human_score = category_scores.get('human_focused', 0.0)
        person_score = tags.get("a photo of a person", 0.0)
        routing['needs_face_detection'] = max(human_score, person_score) > face_threshold

        # Landmark detection
        landmark_threshold = self.routing_thresholds.get('landmark_detection', 0.6)
        landmark_score = tags.get("a photo of a landmark", 0.0)
        routing['needs_landmark_detection'] = landmark_score > landmark_threshold

        # Object detection (optional)
        object_threshold = self.routing_thresholds.get('object_detection', 0.5)
        object_score = max(
            tags.get("a product photo", 0.0),
            tags.get("a vehicle", 0.0)
        )
        routing['needs_object_detection'] = object_score > object_threshold

        # Text/OCR detection
        text_threshold = self.routing_thresholds.get('text_detection', 0.7)
        text_score = tags.get("a document or text page", 0.0)
        routing['needs_text_detection'] = text_score > text_threshold

        return routing

    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze batch of images (optimized with batched CLIP encoding).

        Args:
            image_paths: List of image paths
            batch_size: Batch size for CLIP encoding
            verbose: Whether to show progress

        Returns:
            Dict mapping image paths to analysis results
        """
        results = {}

        # Process in batches for efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            if verbose:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")

            try:
                # Batch encode images
                image_embeddings = self.clip.encode_images(
                    [str(p) for p in batch_paths],
                    batch_size=batch_size
                )

                # Compute similarities for batch
                similarities_batch = self.clip.compute_similarity(
                    image_embeddings,
                    self.prompt_embeddings
                )

                # Process each image in batch
                for j, image_path in enumerate(batch_paths):
                    similarities = similarities_batch[j]

                    # Create tags dict
                    tags = {
                        prompt: float(score)
                        for prompt, score in zip(self.all_prompts, similarities)
                    }

                    # Get top-k tags
                    top_indices = np.argsort(similarities)[::-1][:5]
                    primary_tags = [self.all_prompts[idx] for idx in top_indices]

                    # Aggregate and compute scores
                    category_scores = self._aggregate_by_category(tags)
                    importance_score = self._compute_importance_score(category_scores, tags)
                    routing = self._determine_routing(tags, category_scores)

                    # Store result
                    results[str(image_path)] = {
                        'path': str(image_path),
                        'tags': tags,
                        'primary_tags': primary_tags,
                        'category_scores': category_scores,
                        'importance_score': importance_score,
                        'routing': routing,
                        'metadata': {
                            'total_prompts': self.total_prompts,
                            'top_5_scores': [float(similarities[idx]) for idx in top_indices]
                        }
                    }

                    # Cache if enabled
                    if self.enable_cache:
                        cache_key = str(Path(image_path).resolve())
                        self._analysis_cache[cache_key] = results[str(image_path)]

            except Exception as e:
                logger.error(f"Failed to process batch starting at {i}: {e}")
                # Add empty results for failed batch
                for image_path in batch_paths:
                    results[str(image_path)] = self._get_empty_analysis(image_path)

        if verbose:
            logger.info(f"Analyzed {len(results)} images")

        return results

    def get_prompt_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded prompts.

        Returns:
            Dict with prompt statistics
        """
        return {
            'total_prompts': self.total_prompts,
            'categories': {
                category: len(prompts)
                for category, prompts in self.prompts.items()
            },
            'routing_thresholds': self.routing_thresholds,
            'importance_weights': self.importance_weights
        }

    def __repr__(self) -> str:
        return (
            f"CLIPTagger("
            f"model={self.clip.model_name}, "
            f"prompts={self.total_prompts}, "
            f"device={self.device})"
        )
