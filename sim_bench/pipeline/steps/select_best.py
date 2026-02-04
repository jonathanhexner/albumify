"""Select best step - smart selection with face vs non-face branching logic."""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class SelectBestStep(BaseStep):
    """
    Select best images from each cluster using smart branching logic.

    For face clusters:
        - Use composite score: eyes_open + pose + smile + AVA
        - Configurable weights

    For non-face clusters:
        - Use AVA score as primary
        - Use Siamese CNN as tiebreaker when scores are close

    Smart selection rules:
        - Always take #1
        - Take #2 only if:
            - Score above min threshold
            - Not a near-duplicate of #1 (Siamese CNN check)
            - Score gap not too large
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="select_best",
            display_name="Select Best Images",
            description="Smart selection with branching logic for face vs non-face clusters.",
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
                    "max_score_gap": {
                        "type": "number",
                        "default": 0.25,
                        "description": "Maximum gap between #1 and #2 to keep #2"
                    },
                    "tiebreaker_threshold": {
                        "type": "number",
                        "default": 0.05,
                        "description": "Use Siamese tiebreaker if scores within this range"
                    },
                    "siamese_checkpoint": {
                        "type": "string",
                        "description": "Path to Siamese model checkpoint for comparison"
                    },
                    "duplicate_similarity_threshold": {
                        "type": "number",
                        "default": 0.95,
                        "description": "Embedding similarity threshold for near-duplicates"
                    },
                    "face_weights": {
                        "type": "object",
                        "properties": {
                            "eyes_open": {"type": "number", "default": 0.30},
                            "pose": {"type": "number", "default": 0.30},
                            "smile": {"type": "number", "default": 0.20},
                            "ava": {"type": "number", "default": 0.20}
                        },
                        "description": "Weights for face composite score"
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

    def process(self, context: PipelineContext, config: dict) -> None:
        """Select best images using smart branching logic."""
        max_per_cluster = config.get("max_images_per_cluster", 2)
        min_score_threshold = config.get("min_score_threshold", 0.4)
        max_score_gap = config.get("max_score_gap", 0.25)
        siamese_config = config.get("siamese", {})
        siamese_checkpoint = siamese_config.get("checkpoint_path") if siamese_config.get("enabled", False) else None
        tiebreaker_threshold = siamese_config.get("tiebreaker_range", config.get("tiebreaker_threshold", 0.05))
        duplicate_threshold = siamese_config.get("duplicate_threshold", config.get("duplicate_similarity_threshold", 0.95))
        use_face_subclusters = config.get("use_face_subclusters", True)
        include_noise = config.get("include_noise", True)

        face_weights = config.get("face_weights", {
            "eyes_open": 0.30,
            "pose": 0.30,
            "smile": 0.20,
            "ava": 0.20
        })

        # Load Siamese model if checkpoint provided
        siamese_model = self._get_siamese_model(siamese_checkpoint)
        if siamese_model:
            logger.info("Siamese CNN enabled for comparison and duplicate detection")
        else:
            logger.info("Siamese CNN not available, using embedding similarity")

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
                    has_faces = subcluster.get("has_faces", False)
                    images = subcluster.get("images", [])

                    if not images:
                        continue

                    cluster_selected = self._select_from_cluster(
                        context=context,
                        image_paths=images,
                        has_faces=has_faces,
                        max_per_cluster=max_per_cluster,
                        min_score_threshold=min_score_threshold,
                        max_score_gap=max_score_gap,
                        tiebreaker_threshold=tiebreaker_threshold,
                        duplicate_threshold=duplicate_threshold,
                        face_weights=face_weights,
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

                # Determine if this cluster has faces
                has_faces = self._cluster_has_faces(context, image_paths)

                cluster_selected = self._select_from_cluster(
                    context=context,
                    image_paths=image_paths,
                    has_faces=has_faces,
                    max_per_cluster=max_per_cluster,
                    min_score_threshold=min_score_threshold,
                    max_score_gap=max_score_gap,
                    tiebreaker_threshold=tiebreaker_threshold,
                    duplicate_threshold=duplicate_threshold,
                    face_weights=face_weights,
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
        has_faces: bool,
        max_per_cluster: int,
        min_score_threshold: float,
        max_score_gap: float,
        tiebreaker_threshold: float,
        duplicate_threshold: float,
        face_weights: dict,
        siamese_model
    ) -> list[str]:
        """Select best images from a single cluster."""
        if not image_paths:
            return []

        # Score all images
        if has_faces:
            scored_images = self._score_face_cluster(context, image_paths, face_weights)
        else:
            scored_images = self._score_non_face_cluster(context, image_paths)

        # Persist composite scores in context for database storage
        for path, score in scored_images:
            context.composite_scores[path] = score

        # Sort by score descending
        scored_images.sort(key=lambda x: x[1], reverse=True)

        if not scored_images:
            return []

        # Apply Siamese tiebreaker for top candidates if scores are close
        if siamese_model and len(scored_images) >= 2:
            scored_images = self._apply_siamese_tiebreaker(
                scored_images, tiebreaker_threshold, siamese_model
            )

        # Smart selection logic
        selected = []

        # Always take #1 (if above threshold or we have only one)
        best_path, best_score = scored_images[0]
        if best_score >= min_score_threshold or len(scored_images) == 1:
            selected.append(best_path)

        # Consider #2 if we want more than one
        if max_per_cluster > 1 and len(scored_images) > 1:
            second_path, second_score = scored_images[1]

            # Check smart rules
            should_keep_second = True

            # Rule 1: Score above threshold
            if second_score < min_score_threshold:
                should_keep_second = False
                logger.debug(f"Rejecting {second_path}: score {second_score:.2f} below threshold {min_score_threshold}")

            # Rule 2: Score gap not too large
            if should_keep_second and best_score > 0:
                gap = (best_score - second_score) / best_score
                if gap > max_score_gap:
                    should_keep_second = False
                    logger.debug(f"Rejecting {second_path}: gap {gap:.2f} exceeds max {max_score_gap}")

            # Rule 3: Near-duplicate check using Siamese or embeddings
            if should_keep_second:
                is_duplicate = self._check_near_duplicate(
                    context, best_path, second_path,
                    siamese_model, duplicate_threshold
                )
                if is_duplicate:
                    should_keep_second = False
                    logger.debug(f"Rejecting {second_path}: near-duplicate of {best_path}")

            if should_keep_second:
                selected.append(second_path)

        return selected

    def _apply_siamese_tiebreaker(
        self,
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

            if winner_idx == 1:
                # Swap - second is better
                reranked[0], reranked[1] = reranked[1], reranked[0]
                logger.debug(f"Siamese: {Path(path2).name} beats {Path(path1).name} (conf={result['confidence']:.2f})")

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
        Check if two images are near-duplicates.

        Uses Siamese CNN if available, otherwise falls back to embedding similarity.
        """
        # Try Siamese comparison first
        if siamese_model:
            try:
                result = siamese_model.compare_images(Path(path1), Path(path2))
                # If confidence is very high (>0.95), likely not duplicates
                # If confidence is low (<0.6), they're very similar
                if result['confidence'] < 0.6:
                    logger.debug(f"Siamese duplicate check: {Path(path1).name} vs {Path(path2).name} "
                                f"conf={result['confidence']:.2f} (likely duplicates)")
                    return True
                return False
            except Exception as e:
                logger.warning(f"Siamese duplicate check failed: {e}, falling back to embeddings")

        # Fall back to embedding similarity
        return self._check_embedding_similarity(context, path1, path2, embedding_threshold)

    def _check_embedding_similarity(
        self,
        context: PipelineContext,
        path1: str,
        path2: str,
        threshold: float
    ) -> bool:
        """Check near-duplicate using scene embedding cosine similarity."""
        emb1 = context.scene_embeddings.get(path1)
        emb2 = context.scene_embeddings.get(path2)

        if emb1 is None or emb2 is None:
            return False

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return False

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return similarity > threshold

    def _score_face_cluster(
        self,
        context: PipelineContext,
        image_paths: list[str],
        weights: dict
    ) -> list[tuple[str, float]]:
        """
        Score images in a face cluster using composite score.

        composite = w_eyes * eyes + w_pose * pose + w_smile * smile + w_ava * ava
        """
        scored = []

        for path in image_paths:
            # Get face scores for this image
            # Scores may be keyed by path:face_N format or directly by path
            eyes_scores = self._get_face_scores_for_image(context.face_eyes_scores, path)
            pose_scores = self._get_face_scores_for_image(context.face_pose_scores, path)
            smile_scores = self._get_face_scores_for_image(context.face_smile_scores, path)

            # Average face scores (or default to 0.5 if not available)
            eyes_avg = np.mean(eyes_scores) if eyes_scores else 0.5
            pose_avg = np.mean(pose_scores) if pose_scores else 0.5
            smile_avg = np.mean(smile_scores) if smile_scores else 0.5

            # Get AVA score (normalize to 0-1 if needed)
            ava_score = context.ava_scores.get(path, 5.0)
            if ava_score > 1.0:  # AVA is typically 1-10 scale
                ava_score = ava_score / 10.0

            # Compute weighted composite
            composite = (
                weights.get("eyes_open", 0.30) * eyes_avg +
                weights.get("pose", 0.30) * pose_avg +
                weights.get("smile", 0.20) * smile_avg +
                weights.get("ava", 0.20) * ava_score
            )

            scored.append((path, composite))

        return scored

    def _get_face_scores_for_image(
        self,
        scores_dict: dict,
        image_path: str
    ) -> list[float]:
        """
        Get face scores for an image.

        Handles two formats:
        1. {image_path: [list of scores]} - original format
        2. {image_path:face_N: score} - cache key format
        """
        # Try direct lookup first
        if image_path in scores_dict:
            val = scores_dict[image_path]
            if isinstance(val, list):
                return val
            return [val]

        # Try cache key format (path:face_0, path:face_1, etc.)
        scores = []
        for key, val in scores_dict.items():
            if key.startswith(f"{image_path}:face_"):
                scores.append(val)

        return scores

    def _score_non_face_cluster(
        self,
        context: PipelineContext,
        image_paths: list[str]
    ) -> list[tuple[str, float]]:
        """
        Score images in a non-face cluster using AVA score.
        """
        scored = []

        for path in image_paths:
            # Primary: AVA score
            ava_score = context.ava_scores.get(path, 5.0)
            if ava_score > 1.0:  # AVA is typically 1-10 scale
                ava_score = ava_score / 10.0

            # Secondary: IQA score
            iqa_score = context.iqa_scores.get(path, 0.5)

            # Weighted combination (AVA primary)
            score = 0.7 * ava_score + 0.3 * iqa_score

            scored.append((path, score))

        return scored

    def _cluster_has_faces(self, context: PipelineContext, image_paths: list[str]) -> bool:
        """Check if any image in the cluster has significant faces."""
        for path in image_paths:
            faces = context.faces.get(path, [])
            if faces:
                return True
        return False
