"""Select best step - composite scoring with quality and person penalties."""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.scoring.quality_strategy import (
    ImageQualityStrategyFactory,
    ImageQualityStrategy,
)
from sim_bench.pipeline.scoring.person_penalty import (
    PersonPenaltyFactory,
    PersonPenaltyComputer,
)

logger = logging.getLogger(__name__)


@register_step
class SelectBestStep(BaseStep):
    """
    Select best images from each cluster using composite scoring.

    Scoring Model:
        composite_score = image_quality_score + person_penalty

        image_quality_score: Technical/aesthetic quality (IQA + AVA + optional Siamese)
        person_penalty: Portrait-specific penalties (0 to -0.7)
            - No person: 0 penalty
            - Person with issues: penalties for face occlusion, eyes closed, etc.

    Selection Rules:
        1. Compute composite scores for all images
        2. Filter images below min_threshold
        3. Sort by composite score
        4. Select best + dissimilar images (up to max_per_cluster)
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="select_best",
            display_name="Select Best Images",
            description="Composite scoring with image quality and person penalties.",
            category="selection",
            requires={"scene_clusters"},
            produces={"selected_images"},
            depends_on=["cluster_scenes"],
            config_schema={
                "type": "object",
                "properties": {
                    "max_images_per_cluster": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "description": "Maximum images to select per cluster"
                    },
                    "min_score_threshold": {
                        "type": "number",
                        "default": 0.4,
                        "description": "Minimum composite score to keep an image"
                    },
                    "dissimilarity_threshold": {
                        "type": "number",
                        "default": 0.85,
                        "description": "Embedding similarity threshold for dissimilar images"
                    },
                    "quality_strategy": {
                        "type": "string",
                        "default": "siamese_refinement",
                        "description": "Quality scoring strategy: weighted_average, siamese_refinement, siamese_tournament"
                    },
                    "quality_weights": {
                        "type": "object",
                        "properties": {
                            "iqa": {"type": "number", "default": 0.3},
                            "ava": {"type": "number", "default": 0.7}
                        },
                        "description": "Weights for image quality score"
                    },
                    "siamese_refinement": {
                        "type": "object",
                        "properties": {
                            "top_n": {"type": "integer", "default": 3},
                            "boost_range": {"type": "number", "default": 0.1}
                        },
                        "description": "Config for Siamese refinement strategy"
                    },
                    "person_penalties": {
                        "type": "object",
                        "properties": {
                            "face_occlusion": {"type": "number", "default": -0.3},
                            "body_not_facing": {"type": "number", "default": -0.1},
                            "eyes_closed": {"type": "number", "default": -0.15},
                            "not_smiling": {"type": "number", "default": -0.05},
                            "face_turned": {"type": "number", "default": -0.1},
                            "max_penalty": {"type": "number", "default": -0.7}
                        },
                        "description": "Penalty values for portrait issues"
                    },
                    "siamese": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "checkpoint_path": {"type": "string"}
                        },
                        "description": "Siamese model config"
                    },
                    "use_face_subclusters": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use face subclusters if available"
                    },
                    "include_noise": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include noise cluster (-1) in selection"
                    }
                }
            }
        )
        self._siamese_model = None
        self._siamese_checkpoint = None
        self._config = None
        self._quality_strategy: Optional[ImageQualityStrategy] = None
        self._penalty_computer: Optional[PersonPenaltyComputer] = None

    def _get_siamese_model(self, checkpoint_path: str):
        """Lazy load Siamese model."""
        if checkpoint_path is None:
            return None

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            logger.warning(f"Siamese checkpoint not found: {checkpoint_path}")
            return None

        if self._siamese_model is None or self._siamese_checkpoint != checkpoint_path:
            from sim_bench.image_quality_models.siamese_model_wrapper import SiameseQualityModel
            logger.info(f"Loading Siamese model from {checkpoint_path}")
            self._siamese_model = SiameseQualityModel(checkpoint, device='cpu')
            self._siamese_checkpoint = checkpoint_path

        return self._siamese_model

    def _build_quality_config(self, config: dict, strategy_name: str) -> dict:
        """Build quality strategy config from select_best config."""
        quality_weights = config.get("quality_weights", {"iqa": 0.3, "ava": 0.7})
        quality_config = {
            "iqa_weight": quality_weights.get("iqa", 0.3),
            "ava_weight": quality_weights.get("ava", 0.7),
        }

        strategy_is_refinement = strategy_name == "siamese_refinement"
        if strategy_is_refinement:
            refinement_config = config.get("siamese_refinement", {})
            quality_config["top_n"] = refinement_config.get("top_n", 3)
            quality_config["boost_range"] = refinement_config.get("boost_range", 0.1)

        strategy_is_tournament = strategy_name == "siamese_tournament"
        if strategy_is_tournament:
            tournament_config = config.get("siamese_tournament", {})
            quality_config["top_n"] = tournament_config.get("top_n", 4)
            quality_config["base_weight"] = tournament_config.get("base_weight", 0.4)
            quality_config["siamese_weight"] = tournament_config.get("siamese_weight", 0.6)

        return quality_config

    def process(self, context: PipelineContext, config: dict) -> None:
        """Select best images using composite scoring."""
        self._config = config

        max_per_cluster = config.get("max_images_per_cluster", 2)
        use_face_subclusters = config.get("use_face_subclusters", True)
        include_noise = config.get("include_noise", True)

        # Initialize quality strategy
        quality_strategy_name = config.get("quality_strategy", "siamese_refinement")
        quality_config = self._build_quality_config(config, quality_strategy_name)
        self._quality_strategy = ImageQualityStrategyFactory.create(
            quality_strategy_name, quality_config
        )

        # Initialize person penalty computer
        penalty_config = config.get("person_penalties", {})
        self._penalty_computer = PersonPenaltyFactory.create(penalty_config)

        # Load Siamese model if needed
        siamese_config = config.get("siamese", {})
        siamese_enabled = siamese_config.get("enabled", False)
        siamese_checkpoint = siamese_config.get("checkpoint_path") if siamese_enabled else None
        siamese_model = self._get_siamese_model(siamese_checkpoint)

        strategy_requires_siamese = quality_strategy_name in ["siamese_refinement", "siamese_tournament"]
        if strategy_requires_siamese and not siamese_model:
            logger.warning(
                f"Quality strategy '{quality_strategy_name}' requires Siamese model "
                f"but model not loaded. Falling back to base quality only."
            )

        selected = []
        total_clusters = 0

        # Use face subclusters if available, otherwise use scene clusters
        if use_face_subclusters and context.face_clusters:
            # Process subclusters within each scene
            for scene_id, subclusters in context.face_clusters.items():
                if scene_id == -1 and not include_noise:
                    continue

                for subcluster_id, subcluster in subclusters.items():
                    total_clusters += 1
                    images = subcluster.get("images", [])

                    if not images:
                        continue

                    cluster_selected = self._select_from_cluster(
                        context=context,
                        image_paths=images,
                        max_per_cluster=max_per_cluster,
                        siamese_model=siamese_model
                    )
                    selected.extend(cluster_selected)

                    context.report_progress(
                        "select_best",
                        0.5 + 0.5 * (scene_id + 1) / len(context.face_clusters),
                        f"Processing scene {scene_id}, subcluster {subcluster_id}"
                    )
        else:
            # Fall back to scene clusters
            for cluster_id, image_paths in context.scene_clusters.items():
                if cluster_id == -1 and not include_noise:
                    continue

                total_clusters += 1

                cluster_selected = self._select_from_cluster(
                    context=context,
                    image_paths=image_paths,
                    max_per_cluster=max_per_cluster,
                    siamese_model=siamese_model
                )
                selected.extend(cluster_selected)

                context.report_progress(
                    "select_best",
                    0.5 + 0.5 * (cluster_id + 1) / len(context.scene_clusters),
                    f"Processing cluster {cluster_id}"
                )

        context.selected_images = selected

        context.report_progress(
            "select_best",
            1.0,
            f"Selected {len(selected)} images from {total_clusters} clusters"
        )

    def _select_from_cluster(
        self,
        context: PipelineContext,
        image_paths: list[str],
        max_per_cluster: int,
        siamese_model
    ) -> list[str]:
        """Select best images from a single cluster using composite scoring."""
        if not image_paths:
            return []

        # Compute composite scores for all images
        scored_images = self._compute_composite_scores(
            context, image_paths, siamese_model
        )

        # Persist scores in context
        for path, score in scored_images:
            context.composite_scores[path] = score

        # Filter by minimum threshold
        min_threshold = self._config.get("min_score_threshold", 0.4)
        filtered = [
            (path, score) for path, score in scored_images if score >= min_threshold
        ]

        # Keep best one even if below threshold
        no_images_above_threshold = len(filtered) == 0
        if no_images_above_threshold and scored_images:
            filtered = [max(scored_images, key=lambda x: x[1])]
            logger.debug(
                f"No images above threshold {min_threshold}, "
                f"keeping best: {filtered[0][0]} (score={filtered[0][1]:.3f})"
            )

        # Sort by composite score
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Select best + dissimilar images
        selected = self._select_dissimilar(context, filtered, max_per_cluster)

        return selected

    def _compute_composite_scores(
        self,
        context: PipelineContext,
        image_paths: List[str],
        siamese_model: Optional[object],
    ) -> List[Tuple[str, float]]:
        """
        Compute composite scores for all images.

        composite_score = image_quality_score + person_penalty
        """
        scored = []

        for image_path in image_paths:
            quality_score = self._quality_strategy.compute_quality(
                image_path, context, siamese_model, image_paths
            )
            penalty = self._penalty_computer.compute_penalty(image_path, context)
            composite_score = quality_score + penalty

            scored.append((image_path, composite_score))

        return scored

    def _select_dissimilar(
        self,
        context: PipelineContext,
        scored_images: List[Tuple[str, float]],
        max_per_cluster: int,
    ) -> List[str]:
        """
        Select best + dissimilar images from scored list.

        Selection strategy:
        1. Always select #1 (best)
        2. Select additional images that are sufficiently dissimilar
        """
        if not scored_images:
            return []

        selected = []
        dissimilarity_threshold = self._config.get("dissimilarity_threshold", 0.85)

        # Always select best
        best_path, best_score = scored_images[0]
        selected.append(best_path)

        # Select additional dissimilar images
        for image_path, score in scored_images[1:]:
            max_reached = len(selected) >= max_per_cluster
            if max_reached:
                break

            is_dissimilar = self._check_dissimilar(
                context, image_path, selected, dissimilarity_threshold
            )

            if is_dissimilar:
                selected.append(image_path)
            else:
                logger.debug(
                    f"Skipping {image_path}: too similar to selected images"
                )

        return selected

    def _check_dissimilar(
        self,
        context: PipelineContext,
        image_path: str,
        selected_images: List[str],
        threshold: float,
    ) -> bool:
        """Check if image is sufficiently dissimilar from all selected images."""
        for selected_path in selected_images:
            similarity = self._get_embedding_similarity(
                context, image_path, selected_path
            )

            similarity_available = similarity is not None
            if similarity_available:
                too_similar = similarity >= threshold
                if too_similar:
                    return False

        return True

    def _apply_siamese_tiebreaker(
        self,
        context: PipelineContext,
        scored_images: List[Tuple[str, float]],
        threshold: float,
        siamese_model
    ) -> List[Tuple[str, float]]:
        """
        Apply Siamese CNN to break ties when scores are close.

        If top candidates have scores within threshold, use Siamese to
        determine the actual winner.
        """
        if len(scored_images) < 2:
            return scored_images

        # Check if top scores are within tiebreaker threshold
        top_score = scored_images[0][1]
        candidates_to_compare = []

        for path, score in scored_images[:3]:  # Compare top 3 at most
            if abs(score - top_score) <= threshold:
                candidates_to_compare.append((path, score))
            else:
                break

        if len(candidates_to_compare) < 2:
            return scored_images

        logger.debug(f"Siamese tiebreaker: comparing {len(candidates_to_compare)} candidates")

        # Run pairwise Siamese comparisons
        # Simple tournament: compare adjacent pairs and keep winner
        reranked = list(candidates_to_compare)

        try:
            # Compare first vs second
            path1, score1 = reranked[0]
            path2, score2 = reranked[1]

            result = siamese_model.compare_images(Path(path1), Path(path2))
            winner_idx = 0 if result['prediction'] == 1 else 1
            winner_path = path1 if winner_idx == 0 else path2
            confidence = result['confidence']

            # Log comparison for visibility
            context.siamese_comparisons.append({
                'type': 'tiebreaker',
                'img1': Path(path1).name,
                'img2': Path(path2).name,
                'img1_path': path1,
                'img2_path': path2,
                'winner': Path(winner_path).name,
                'confidence': round(confidence, 3),
                'score1': round(score1, 3),
                'score2': round(score2, 3),
            })

            if winner_idx == 1:
                # Swap - second is better
                reranked[0], reranked[1] = reranked[1], reranked[0]
                logger.debug(f"Siamese: {Path(path2).name} beats {Path(path1).name} (conf={confidence:.2f})")

            # Rebuild full list with reranked top
            result_list = reranked + [x for x in scored_images if x not in candidates_to_compare]
            return result_list

        except Exception as e:
            logger.warning(f"Siamese comparison failed: {e}")
            return scored_images

    def _check_near_duplicate(
        self,
        context: PipelineContext,
        path1: str,
        path2: str,
        siamese_model,
        embedding_threshold: float
    ) -> bool:
        """
        Check if two images are near-duplicates using embedding similarity.

        NOTE: Siamese CNN is NOT used for duplicate detection because it compares
        image QUALITY (which is better), not image SIMILARITY (are they the same).
        Low Siamese confidence means it can't tell which is better quality,
        not that they are duplicates.
        """
        # Always use embedding similarity for duplicate detection
        similarity = self._get_embedding_similarity(context, path1, path2)
        is_duplicate = similarity > embedding_threshold if similarity is not None else False

        # Log comparison for visibility
        context.siamese_comparisons.append({
            'type': 'duplicate_check',
            'img1': Path(path1).name,
            'img2': Path(path2).name,
            'img1_path': path1,
            'img2_path': path2,
            'is_duplicate': is_duplicate,
            'method': 'embedding',
            'similarity': round(similarity, 3) if similarity else None,
            'threshold': embedding_threshold,
        })

        if is_duplicate:
            logger.debug(f"Duplicate check: {Path(path1).name} vs {Path(path2).name} "
                        f"similarity={similarity:.3f} > threshold={embedding_threshold}")

        return is_duplicate

    def _get_embedding_similarity(
        self,
        context: PipelineContext,
        path1: str,
        path2: str
    ) -> Optional[float]:
        """Get cosine similarity between scene embeddings."""
        emb1 = context.scene_embeddings.get(path1)
        emb2 = context.scene_embeddings.get(path2)

        if emb1 is None or emb2 is None:
            return None

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return None

        return float(np.dot(emb1, emb2) / (norm1 * norm2))