"""Image quality scoring strategies."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ImageQualityStrategy(ABC):
    """Abstract base class for image quality scoring strategies."""

    @abstractmethod
    def compute_quality(
        self,
        image_path: str,
        context: PipelineContext,
        siamese_model: Optional[object],
        all_images: List[str],
    ) -> float:
        """
        Compute quality score for image.

        Args:
            image_path: Path to the image
            context: Pipeline context with IQA, AVA scores
            siamese_model: Optional Siamese CNN model
            all_images: All images in the cluster

        Returns:
            Quality score in [0, 1]
        """
        pass


class WeightedAverageQuality(ImageQualityStrategy):
    """Simple weighted average of IQA and AVA scores."""

    def __init__(self, iqa_weight: float = 0.3, ava_weight: float = 0.7):
        """
        Initialize weighted average strategy.

        Args:
            iqa_weight: Weight for IQA score
            ava_weight: Weight for AVA score
        """
        self.iqa_weight = iqa_weight
        self.ava_weight = ava_weight
        logger.info(f"WeightedAverageQuality: IQA={iqa_weight}, AVA={ava_weight}")

    def compute_quality(
        self,
        image_path: str,
        context: PipelineContext,
        siamese_model: Optional[object],
        all_images: List[str],
    ) -> float:
        """Compute weighted average of IQA and AVA."""
        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)

        quality = self.iqa_weight * iqa + self.ava_weight * ava
        return quality


class SiameseRefinementQuality(ImageQualityStrategy):
    """Refine top candidates with Siamese comparisons against best image."""

    def __init__(
        self,
        iqa_weight: float = 0.3,
        ava_weight: float = 0.7,
        top_n: int = 3,
        boost_range: float = 0.1,
    ):
        """
        Initialize Siamese refinement strategy.

        Args:
            iqa_weight: Weight for IQA score
            ava_weight: Weight for AVA score
            top_n: Number of top candidates to refine
            boost_range: Range for Siamese boost/penalty (±)
        """
        self.iqa_weight = iqa_weight
        self.ava_weight = ava_weight
        self.top_n = top_n
        self.boost_range = boost_range
        self._base_quality_cache: Dict[str, float] = {}
        logger.info(
            f"SiameseRefinementQuality: IQA={iqa_weight}, AVA={ava_weight}, "
            f"top_n={top_n}, boost=±{boost_range}"
        )

    def compute_quality(
        self,
        image_path: str,
        context: PipelineContext,
        siamese_model: Optional[object],
        all_images: List[str],
    ) -> float:
        """Compute quality with Siamese refinement for top candidates."""
        base_quality = self._compute_base_quality(image_path, context)

        quality_without_siamese = siamese_model is None
        if quality_without_siamese:
            return base_quality

        top_candidates = self._get_top_n_by_base_quality(all_images, context)
        not_in_top = image_path not in top_candidates

        if not_in_top:
            return base_quality

        return self._apply_siamese_refinement(
            image_path, top_candidates, siamese_model, base_quality, context
        )

    def _compute_base_quality(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute base quality from IQA and AVA."""
        cached = self._base_quality_cache.get(image_path)
        if cached is not None:
            return cached

        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)

        ava_out_of_range = ava > 1.0
        if ava_out_of_range:
            logger.warning(f"AVA score for {image_path} is {ava} (>1.0)")

        base_quality = self.iqa_weight * iqa + self.ava_weight * ava
        self._base_quality_cache[image_path] = base_quality
        return base_quality

    def _get_top_n_by_base_quality(
        self, all_images: List[str], context: PipelineContext
    ) -> List[str]:
        """Get top N images by base quality."""
        scored = [
            (img, self._compute_base_quality(img, context)) for img in all_images
        ]
        sorted_images = sorted(scored, key=lambda x: x[1], reverse=True)
        top_n = [img for img, _ in sorted_images[: self.top_n]]
        return top_n

    def _apply_siamese_refinement(
        self,
        image_path: str,
        top_candidates: List[str],
        siamese_model: object,
        base_quality: float,
        context: PipelineContext,
    ) -> float:
        """Apply Siamese refinement for top candidate."""
        from pathlib import Path

        best_image = top_candidates[0]

        is_best = image_path == best_image
        if is_best:
            return base_quality

        result = siamese_model.compare_images(image_path, best_image)
        current_is_better = result["prediction"] == 1
        confidence = result["confidence"]

        # Log comparison for visibility
        context.siamese_comparisons.append({
            'type': 'refinement',
            'img1': Path(image_path).name,
            'img2': Path(best_image).name,
            'img1_path': image_path,
            'img2_path': best_image,
            'winner': Path(image_path).name if current_is_better else Path(best_image).name,
            'confidence': round(confidence, 3),
            'base_quality': round(base_quality, 3),
        })

        boost_multiplier = 1.0 if current_is_better else -1.0
        boost = boost_multiplier * self.boost_range * confidence

        refined_quality = base_quality + boost
        return refined_quality


class SiameseTournamentQuality(ImageQualityStrategy):
    """Run Siamese tournament among top candidates (all pairwise comparisons)."""

    def __init__(
        self,
        iqa_weight: float = 0.3,
        ava_weight: float = 0.7,
        top_n: int = 4,
        base_weight: float = 0.4,
        siamese_weight: float = 0.6,
    ):
        """
        Initialize Siamese tournament strategy.

        Args:
            iqa_weight: Weight for IQA score
            ava_weight: Weight for AVA score
            top_n: Number of top candidates for tournament
            base_weight: Weight for base quality in final score
            siamese_weight: Weight for Siamese score in final score
        """
        self.iqa_weight = iqa_weight
        self.ava_weight = ava_weight
        self.top_n = top_n
        self.base_weight = base_weight
        self.siamese_weight = siamese_weight
        self._base_quality_cache: Dict[str, float] = {}
        logger.info(
            f"SiameseTournamentQuality: IQA={iqa_weight}, AVA={ava_weight}, "
            f"top_n={top_n}, base_weight={base_weight}, siamese_weight={siamese_weight}"
        )

    def compute_quality(
        self,
        image_path: str,
        context: PipelineContext,
        siamese_model: Optional[object],
        all_images: List[str],
    ) -> float:
        """Compute quality with Siamese tournament for top candidates."""
        base_quality = self._compute_base_quality(image_path, context)

        quality_without_siamese = siamese_model is None
        if quality_without_siamese:
            return base_quality

        top_candidates = self._get_top_n_by_base_quality(all_images, context)
        not_in_top = image_path not in top_candidates

        if not_in_top:
            return base_quality

        siamese_score = self._run_tournament(
            image_path, top_candidates, siamese_model, context
        )
        final_quality = (
            self.base_weight * base_quality + self.siamese_weight * siamese_score
        )
        return final_quality

    def _compute_base_quality(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute base quality from IQA and AVA."""
        cached = self._base_quality_cache.get(image_path)
        if cached is not None:
            return cached

        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)

        ava_out_of_range = ava > 1.0
        if ava_out_of_range:
            logger.warning(f"AVA score for {image_path} is {ava} (>1.0)")

        base_quality = self.iqa_weight * iqa + self.ava_weight * ava
        self._base_quality_cache[image_path] = base_quality
        return base_quality

    def _get_top_n_by_base_quality(
        self, all_images: List[str], context: PipelineContext
    ) -> List[str]:
        """Get top N images by base quality."""
        scored = [
            (img, self._compute_base_quality(img, context)) for img in all_images
        ]
        sorted_images = sorted(scored, key=lambda x: x[1], reverse=True)
        top_n = [img for img, _ in sorted_images[: self.top_n]]
        return top_n

    def _run_tournament(
        self,
        image_path: str,
        top_candidates: List[str],
        siamese_model: object,
        context: PipelineContext,
    ) -> float:
        """Run pairwise tournament and compute win rate."""
        from pathlib import Path

        total_wins = 0.0
        num_comparisons = 0

        for opponent in top_candidates:
            is_self_comparison = image_path == opponent
            if is_self_comparison:
                continue

            result = siamese_model.compare_images(image_path, opponent)
            current_is_better = result["prediction"] == 1
            confidence = result["confidence"]

            # Log comparison for visibility
            context.siamese_comparisons.append({
                'type': 'tournament',
                'img1': Path(image_path).name,
                'img2': Path(opponent).name,
                'img1_path': image_path,
                'img2_path': opponent,
                'winner': Path(image_path).name if current_is_better else Path(opponent).name,
                'confidence': round(confidence, 3),
            })

            if current_is_better:
                total_wins += confidence

            num_comparisons += 1

        no_comparisons = num_comparisons == 0
        if no_comparisons:
            return 0.5

        win_rate = total_wins / num_comparisons
        return win_rate


class ImageQualityStrategyFactory:
    """Factory for creating image quality strategies."""

    @staticmethod
    def create(strategy_name: str, config: Dict) -> ImageQualityStrategy:
        """
        Create image quality strategy from config.

        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary

        Returns:
            ImageQualityStrategy instance
        """
        strategy_map = {
            "weighted_average": WeightedAverageQuality,
            "siamese_refinement": SiameseRefinementQuality,
            "siamese_tournament": SiameseTournamentQuality,
        }

        strategy_class = strategy_map.get(strategy_name)
        unknown_strategy = strategy_class is None

        if unknown_strategy:
            logger.warning(
                f"Unknown quality strategy '{strategy_name}', "
                f"defaulting to 'siamese_refinement'"
            )
            strategy_class = SiameseRefinementQuality

        strategy_instance = strategy_class(**config)
        return strategy_instance
