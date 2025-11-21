"""
CLIP-based quality assessment for specific attributes.

Each method focuses on a single quality attribute using contrastive prompts.
NO aggregation - each attribute gets its own registered method.

This follows the user requirement:
"Each CLIP attribute must be SEPARATE (NO aggregation)"
"""

from typing import Dict, Any
from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.registry import register_method


# Attribute-specific prompt pairs (positive, negative)
ATTRIBUTE_PROMPTS = {
    'aesthetic_overall': [
        ("a high quality photograph", "a low quality photograph"),
        ("a beautiful image", "an ugly image"),
        ("professional photography", "amateur snapshot"),
    ],
    'composition': [
        ("a well-composed photograph", "a poorly-composed photograph"),
        ("good composition and framing", "bad composition and framing"),
        ("balanced visual arrangement", "unbalanced visual arrangement"),
    ],
    'subject_placement': [
        ("a photo with the subject well placed in the frame", "a photo with the subject not well placed in the frame"),
        ("good subject positioning", "poor subject positioning"),
        ("well-centered subject", "off-center awkward subject placement"),
    ],
    'cropping': [
        ("a photo that is well cropped", "a photo that is poorly cropped"),
        ("appropriate crop and framing", "inappropriate crop and framing"),
        ("complete subject in frame", "subject cut off or incomplete"),
    ],
    'sharpness': [
        ("a sharp, in-focus photograph", "a blurry, out-of-focus photograph"),
        ("crisp and clear image", "fuzzy and unclear image"),
        ("high sharpness and detail", "low sharpness and detail"),
    ],
    'exposure': [
        ("a well-exposed photograph", "a poorly-exposed photograph"),
        ("good lighting and brightness", "bad lighting and brightness"),
        ("correct exposure levels", "incorrect exposure levels"),
    ],
    'color': [
        ("vibrant, natural colors", "dull, unnatural colors"),
        ("good color quality", "poor color quality"),
        ("pleasing color palette", "unpleasing color palette"),
    ],
}


def create_clip_attribute_method(attribute_name: str, prompts: list):
    """
    Factory function to create attribute-specific CLIP assessor classes.

    Args:
        attribute_name: Name of the quality attribute
        prompts: List of (positive, negative) prompt pairs

    Returns:
        Class for assessing this specific attribute
    """

    class CLIPAttributeAssessor(QualityAssessor):
        f"""
        CLIP-based assessment for {attribute_name}.

        Uses contrastive prompts specific to this attribute.
        Score = mean(similarity(image, positive) - similarity(image, negative))
        """

        def __init__(
            self,
            model_name: str = "ViT-B-32",
            pretrained: str = "laion2b_s34b_b79k",
            device: str = "cpu",
            enable_cache: bool = True
        ):
            """
            Initialize CLIP attribute assessor.

            Args:
                model_name: OpenCLIP model architecture
                pretrained: Pretrained checkpoint
                device: Device to run on
                enable_cache: Whether to cache scores
            """
            super().__init__(device=device, enable_cache=enable_cache)

            self.model_name = model_name
            self.pretrained = pretrained
            self.attribute_name = attribute_name
            self.prompts = prompts

            # Import here to avoid circular dependencies
            try:
                from sim_bench.vision_language.clip import CLIPModel

                self.clip_model = CLIPModel(
                    model_name=model_name,
                    pretrained=pretrained,
                    device=device,
                    enable_cache=enable_cache
                )

                # Pre-encode all prompts
                self._encode_prompts()

            except ImportError as e:
                raise ImportError(
                    "CLIP requires PyTorch and OpenCLIP. "
                    "Install with: pip install torch open-clip-torch\n"
                    f"Error: {e}"
                )

        def _encode_prompts(self):
            """Pre-encode all prompts for efficiency."""
            all_prompts = []
            for pos, neg in self.prompts:
                all_prompts.extend([pos, neg])

            self.prompt_embeddings = self.clip_model.encode_texts(all_prompts)

        @classmethod
        def is_available(cls) -> bool:
            """Check if CLIP dependencies are available."""
            try:
                import torch
                import open_clip
                return True
            except ImportError:
                return False

        @classmethod
        def from_config(cls, config: Dict) -> 'CLIPAttributeAssessor':
            """Create from config dict."""
            return cls(
                model_name=config.get('model_name', 'ViT-B-32'),
                pretrained=config.get('pretrained', 'laion2b_s34b_b79k'),
                device=config.get('device', 'cpu'),
                enable_cache=config.get('enable_cache', True)
            )

        def assess_image(self, image_path: str) -> float:
            """
            Assess image quality for this specific attribute.

            Args:
                image_path: Path to image

            Returns:
                Quality score (higher is better, typically in range [-1, 1])
            """
            # Check cache
            if self.enable_cache and image_path in self._score_cache:
                return self._score_cache[image_path]

            # Encode image (CLIPModel.encode_images expects a list)
            image_embeddings = self.clip_model.encode_images([image_path])
            image_embedding = image_embeddings[0]

            # Compute contrastive scores for each prompt pair
            scores = []
            prompt_idx = 0

            for pos_prompt, neg_prompt in self.prompts:
                pos_emb = self.prompt_embeddings[prompt_idx]
                neg_emb = self.prompt_embeddings[prompt_idx + 1]
                prompt_idx += 2

                # Cosine similarity
                sim_pos = float(image_embedding @ pos_emb)
                sim_neg = float(image_embedding @ neg_emb)

                # Contrastive score
                contrastive_score = sim_pos - sim_neg
                scores.append(contrastive_score)

            # Average across all prompt pairs
            final_score = float(sum(scores) / len(scores))

            # Cache result
            if self.enable_cache:
                self._score_cache[image_path] = final_score

            return final_score

        def get_config(self) -> Dict[str, Any]:
            """Get method configuration."""
            config = super().get_config()
            config.update({
                'attribute': self.attribute_name,
                'model_name': self.model_name,
                'pretrained': self.pretrained,
                'prompts': self.prompts
            })
            return config

        def get_method_name(self) -> str:
            """Return method name for reporting."""
            return f"CLIP_{self.attribute_name}"

        def __repr__(self) -> str:
            return (
                f"CLIP{self.attribute_name.title()}Assessor("
                f"model='{self.model_name}', "
                f"pretrained='{self.pretrained}')"
            )

    return CLIPAttributeAssessor


# Create and register all attribute-specific methods
for attr_name, prompts in ATTRIBUTE_PROMPTS.items():
    # Create class
    cls = create_clip_attribute_method(attr_name, prompts)

    # Register with registry
    # Method name format: clip_aesthetic_overall, clip_composition, etc.
    method_name = f"clip_{attr_name}"
    register_method(method_name)(cls)


__all__ = ['ATTRIBUTE_PROMPTS', 'create_clip_attribute_method']
