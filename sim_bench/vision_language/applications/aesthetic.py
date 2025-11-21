"""
Aesthetic quality assessment using vision-language models.

Uses contrastive text prompts to evaluate image aesthetics across
multiple dimensions: composition, framing, cropping, overall quality.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from sim_bench.vision_language.base import BaseVisionLanguageModel


class AestheticAssessor:
    """
    Assess image aesthetic quality using vision-language models.

    Uses contrastive text prompts to evaluate composition,
    framing, and overall quality. Returns aggregated aesthetic scores.

    Example:
        >>> from sim_bench.vision_language import CLIPModel
        >>> from sim_bench.vision_language.applications import AestheticAssessor
        >>>
        >>> clip = CLIPModel("ViT-B-32", device="cuda")
        >>> assessor = AestheticAssessor(clip)
        >>> score = assessor.assess_image("photo.jpg")
    """

    # Contrastive prompt pairs (positive, negative)
    CONTRASTIVE_PROMPTS = [
        ("a well-composed photograph", "a poorly-composed photograph"),
        ("a photo with the subject well placed in the frame",
         "a photo with the subject not well placed in the frame"),
        ("a photo that is well cropped", "a photo that is poorly cropped"),
        ("Good Quality photo", "Poor Quality photo"),
    ]

    # Positive attributes (higher similarity = better)
    POSITIVE_PROMPTS = [
        "professional photography",
        "aesthetically pleasing",
    ]

    # Negative attributes (lower similarity = better)
    NEGATIVE_PROMPTS = [
        "amateur snapshot",
        "poor framing",
    ]

    def __init__(
        self,
        model: BaseVisionLanguageModel,
        aggregation: str = "weighted",
        custom_prompts: Optional[Dict[str, List]] = None
    ):
        """
        Initialize aesthetic assessor.

        Args:
            model: Vision-language model instance
            aggregation: Aggregation method ('weighted', 'contrastive_only', 'mean')
            custom_prompts: Optional custom prompts dict with keys:
                'contrastive', 'positive', 'negative'
        """
        self.model = model
        self.aggregation = aggregation

        # Allow custom prompts
        if custom_prompts:
            self.contrastive_prompts = custom_prompts.get(
                'contrastive', self.CONTRASTIVE_PROMPTS
            )
            self.positive_prompts = custom_prompts.get(
                'positive', self.POSITIVE_PROMPTS
            )
            self.negative_prompts = custom_prompts.get(
                'negative', self.NEGATIVE_PROMPTS
            )
        else:
            self.contrastive_prompts = self.CONTRASTIVE_PROMPTS
            self.positive_prompts = self.POSITIVE_PROMPTS
            self.negative_prompts = self.NEGATIVE_PROMPTS

        # Pre-encode all prompts
        self._encode_prompts()

    def _encode_prompts(self):
        """Pre-encode all aesthetic prompts once."""
        all_prompts = []

        # Flatten contrastive pairs
        for pos, neg in self.contrastive_prompts:
            all_prompts.extend([pos, neg])

        all_prompts.extend(self.positive_prompts)
        all_prompts.extend(self.negative_prompts)

        self.prompt_embeddings = self.model.encode_texts(all_prompts)

        # Store counts for slicing
        self.n_contrasts = len(self.contrastive_prompts)
        self.n_positive = len(self.positive_prompts)
        self.n_negative = len(self.negative_prompts)

    def assess_image(self, image_path: str) -> float:
        """
        Assess aesthetic quality of single image.

        Args:
            image_path: Path to image file

        Returns:
            Aesthetic score (higher = better quality)
        """
        return self.assess_batch([image_path])[0]

    def assess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Assess aesthetic quality of multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            Array of aesthetic scores [n_images]
        """
        # Encode images
        image_embs = self.model.encode_images(image_paths)

        # Compute similarities with all prompts
        similarities = self.model.compute_similarity(
            image_embs,
            self.prompt_embeddings
        )

        # Aggregate based on method
        if self.aggregation == "contrastive_only":
            scores = self._aggregate_contrastive(similarities)
        elif self.aggregation == "weighted":
            scores = self._aggregate_weighted(similarities)
        else:  # mean
            scores = np.mean(similarities, axis=1)

        return scores

    def assess_with_details(
        self,
        image_path: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Assess image with detailed score breakdown.

        Args:
            image_path: Path to image file

        Returns:
            (overall_score, detailed_scores_dict)
        """
        # Encode image
        image_emb = self.model.encode_images([image_path])[0]

        # Compute similarities
        similarities = self.model.compute_similarity(
            image_emb.reshape(1, -1),
            self.prompt_embeddings
        )[0]

        # Build detailed scores
        detailed = {}
        idx = 0

        # Contrastive pairs
        for i, (pos_text, neg_text) in enumerate(self.contrastive_prompts):
            pos_sim = similarities[idx]
            neg_sim = similarities[idx + 1]
            contrast_score = pos_sim - neg_sim

            detailed[f"contrast_{i}_{pos_text[:30]}"] = contrast_score
            detailed[f"pos_{i}_{pos_text[:30]}"] = pos_sim
            detailed[f"neg_{i}_{neg_text[:30]}"] = neg_sim

            idx += 2

        # Positive attributes
        for i, text in enumerate(self.positive_prompts):
            detailed[f"positive_{i}_{text[:30]}"] = similarities[idx]
            idx += 1

        # Negative attributes
        for i, text in enumerate(self.negative_prompts):
            detailed[f"negative_{i}_{text[:30]}"] = similarities[idx]
            idx += 1

        # Overall score
        overall = self.assess_batch([image_path])[0]
        detailed['overall'] = overall

        return overall, detailed

    def _aggregate_contrastive(self, similarities: np.ndarray) -> np.ndarray:
        """Aggregate using only contrastive pairs."""
        contrastive_scores = []

        for i in range(self.n_contrasts):
            pos_sim = similarities[:, i*2]
            neg_sim = similarities[:, i*2 + 1]
            contrastive_scores.append(pos_sim - neg_sim)

        return np.mean(contrastive_scores, axis=0)

    def _aggregate_weighted(self, similarities: np.ndarray) -> np.ndarray:
        """Weighted aggregation of all prompts."""
        # Contrastive component
        contrastive = self._aggregate_contrastive(similarities)

        # Positive component
        start_idx = self.n_contrasts * 2
        end_idx = start_idx + self.n_positive
        positive = np.mean(similarities[:, start_idx:end_idx], axis=1)

        # Negative component (inverted: lower similarity = better)
        start_idx = end_idx
        end_idx = start_idx + self.n_negative
        negative = -np.mean(similarities[:, start_idx:end_idx], axis=1)

        # Weighted combination
        return 0.5 * contrastive + 0.3 * positive + 0.2 * negative

    def get_prompt_summary(self) -> Dict[str, int]:
        """Get summary of configured prompts."""
        return {
            'contrastive_pairs': self.n_contrasts,
            'positive_attributes': self.n_positive,
            'negative_attributes': self.n_negative,
            'total_prompts': len(self.prompt_embeddings)
        }
