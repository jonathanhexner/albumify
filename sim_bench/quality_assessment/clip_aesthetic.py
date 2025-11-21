"""
CLIP-based aesthetic assessment using text prompt similarity.

REFACTORED to use sim_bench.vision_language architecture.
This is now a thin wrapper that maintains backward compatibility
while leveraging the new modular vision-language system.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import yaml

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.registry import register_method


@register_method('clip_aesthetic')
class CLIPAestheticAssessor(QualityAssessor):
    """
    CLIP-based aesthetic quality assessment using text prompts.

    **Architecture**: This class is now a thin wrapper over:
    - sim_bench.vision_language.clip.CLIPModel (CLIP implementation)
    - sim_bench.vision_language.applications.aesthetic.AestheticAssessor (aesthetic logic)

    **Benefits of new architecture**:
    - No code duplication with feature_extraction/openclip.py
    - Reusable CLIP model for multiple tasks (quality, retrieval, classification)
    - Cleaner separation of concerns
    - Easier to extend with new VL models (BLIP, LLaVA, etc.)

    **Backward compatibility**: This API remains unchanged, existing code continues to work.

    Example:
        >>> assessor = CLIPAestheticAssessor(
        >>>     model_name="ViT-B-32",
        >>>     device="cuda",
        >>>     aggregation_method="weighted"
        >>> )
        >>> score = assessor.assess_image("photo.jpg")
        >>> detailed = assessor.get_detailed_scores("photo.jpg")
    """

    # Class-level prompt definitions (for backward compatibility)
    CONTRASTIVE_PAIRS = [
        ("a well-composed photograph", "a poorly-composed photograph"),
        ("a photo with the subject well placed in the frame",
         "a photo with the subject not well placed in the frame"),
        ("a photo that is well cropped", "a photo that is poorly cropped"),
        ("Good Quality photo", "Poor Quality photo"),
    ]

    POSITIVE_ATTRIBUTES = [
        "professional photography",
        "aesthetically pleasing",
    ]

    NEGATIVE_ATTRIBUTES = [
        "amateur snapshot",
        "poor framing",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        enable_cache: bool = True,
        aggregation_method: str = "weighted",
    ):
        """
        Initialize CLIP aesthetic assessor.

        Args:
            model_name: OpenCLIP model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained checkpoint (e.g., 'laion2b_s34b_b79k')
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache quality scores
            aggregation_method: How to combine scores:
                - 'weighted': Weighted average (contrastive pairs emphasized)
                - 'contrastive_only': Only use contrastive pairs
                - 'mean': Simple average of all scores
        """
        super().__init__(device=device, enable_cache=enable_cache)

        self.model_name = model_name
        self.pretrained = pretrained
        self.aggregation_method = aggregation_method

        # Import here to avoid circular dependencies and graceful degradation
        try:
            from sim_bench.vision_language.clip import CLIPModel
            from sim_bench.vision_language.applications.aesthetic import AestheticAssessor

            # Create CLIP model
            self.clip_model = CLIPModel(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                enable_cache=enable_cache
            )

            # Create aesthetic assessor with CLIP model
            # Use class-level prompts for consistency
            custom_prompts = {
                'contrastive': self.CONTRASTIVE_PAIRS,
                'positive': self.POSITIVE_ATTRIBUTES,
                'negative': self.NEGATIVE_ATTRIBUTES
            }

            self.aesthetic_assessor = AestheticAssessor(
                model=self.clip_model,
                aggregation=aggregation_method,
                custom_prompts=custom_prompts
            )

        except ImportError as e:
            raise ImportError(
                "CLIP Aesthetic requires PyTorch and OpenCLIP. "
                "Install with: pip install torch open-clip-torch\n"
                f"Error: {e}"
            )

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
    def from_config(cls, config: Dict) -> 'CLIPAestheticAssessor':
        """
        Create CLIPAestheticAssessor from config dict.

        Handles multiple config formats:
        - variant: 'laion', 'sac', 'openai', 'learned'
        - pretrained: explicit checkpoint name
        - prompts_config: path to learned prompts YAML (for 'learned' variant)
        - custom_prompts: dict with 'contrastive', 'positive', 'negative' lists (for attribute-specific prompts)

        Args:
            config: Configuration dictionary

        Returns:
            Configured CLIPAestheticAssessor instance
        """
        # Map variant names to pretrained checkpoints
        # Note: laion2b_s34b_b88k is for ViT-B-16, not ViT-B-32
        # For ViT-B-32, use laion2b_s34b_b79k
        pretrained_map = {
            'laion': 'laion2b_s34b_b79k',
            'sac': 'laion2b_s34b_b79k',  # Fixed: use valid checkpoint for ViT-B-32
            'openai': 'openai'
        }

        # Extract variant or use pretrained directly
        variant = config.get('variant', 'laion')

        # Handle custom prompts (for attribute-specific evaluation)
        custom_prompts = config.get('custom_prompts')
        if custom_prompts:
            # Create instance with custom prompts
            pretrained = config.get('pretrained', pretrained_map.get(variant, 'laion2b_s34b_b79k'))
            instance = cls(
                model_name=config.get('model_name', 'ViT-B-32'),
                pretrained=pretrained,
                device=config.get('device', 'cpu'),
                enable_cache=config.get('enable_cache', True),
                aggregation_method=config.get('aggregation_method', 'contrastive_only')  # Use contrastive_only for single attribute
            )
            
            # Override prompts with custom ones
            if 'contrastive' in custom_prompts:
                # Convert to list of tuples
                contrastive_pairs = custom_prompts['contrastive']
                if isinstance(contrastive_pairs, list) and len(contrastive_pairs) > 0:
                    if isinstance(contrastive_pairs[0], (list, tuple)):
                        # List of [positive, negative] pairs
                        contrastive_tuples = [tuple(pair) if isinstance(pair, list) else pair for pair in contrastive_pairs]
                    elif len(contrastive_pairs) == 2:
                        # Single pair as [positive, negative]
                        contrastive_tuples = [(contrastive_pairs[0], contrastive_pairs[1])]
                    else:
                        raise ValueError("contrastive must be a list of [positive, negative] pairs or a single [positive, negative] pair")
                    
                    instance.aesthetic_assessor.contrastive_prompts = contrastive_tuples
                    instance.aesthetic_assessor.positive_prompts = custom_prompts.get('positive', [])
                    instance.aesthetic_assessor.negative_prompts = custom_prompts.get('negative', [])
                    # Re-encode prompts
                    instance.aesthetic_assessor._encode_prompts()
            
            return instance

        # Handle 'learned' variant (uses custom prompts from file)
        if variant == 'learned':
            pretrained = config.get('pretrained', 'laion2b_s34b_b79k')
            prompts_config = config.get('prompts_config')

            # Create instance with learned prompts
            instance = cls(
                model_name=config.get('model_name', 'ViT-B-32'),
                pretrained=pretrained,
                device=config.get('device', 'cpu'),
                enable_cache=config.get('enable_cache', True),
                aggregation_method=config.get('aggregation_method', 'weighted')
            )

            # Load custom prompts if provided
            if prompts_config:
                instance._load_learned_prompts(prompts_config)

            return instance

        # Standard variants
        pretrained = config.get('pretrained', pretrained_map.get(variant, 'laion2b_s34b_b79k'))

        return cls(
            model_name=config.get('model_name', 'ViT-B-32'),
            pretrained=pretrained,
            device=config.get('device', 'cpu'),
            enable_cache=config.get('enable_cache', True),
            aggregation_method=config.get('aggregation_method', 'weighted')
        )

    def _load_learned_prompts(self, prompts_config_path: str):
        """Load learned prompts from YAML config."""
        try:
            with open(prompts_config_path, 'r') as f:
                prompts_data = yaml.safe_load(f)

            # Update prompts in aesthetic assessor
            if 'contrastive_pairs' in prompts_data:
                self.aesthetic_assessor.contrastive_pairs = prompts_data['contrastive_pairs']
            if 'positive_attributes' in prompts_data:
                self.aesthetic_assessor.positive_attributes = prompts_data['positive_attributes']
            if 'negative_attributes' in prompts_data:
                self.aesthetic_assessor.negative_attributes = prompts_data['negative_attributes']

        except Exception as e:
            print(f"Warning: Could not load learned prompts from {prompts_config_path}: {e}")

    def assess_image(self, image_path: str) -> float:
        """
        Assess aesthetic quality of single image.

        Args:
            image_path: Path to image file

        Returns:
            Quality score (higher is better, typically in range [-1, 1])
        """
        # Check cache
        if self.enable_cache and image_path in self._score_cache:
            return self._score_cache[image_path]

        # Assess using underlying aesthetic assessor
        score = self.aesthetic_assessor.assess_image(image_path)

        # Cache result
        if self.enable_cache:
            self._score_cache[image_path] = score

        return score

    def get_detailed_scores(self, image_path: str) -> Optional[Dict[str, float]]:
        """
        Get detailed aesthetic dimension scores for an image.

        Returns dict with:
        - 'overall': Overall aesthetic score
        - 'contrast_N_*': Contrastive pair scores
        - 'pos_N_*': Positive attribute scores
        - 'neg_N_*': Negative attribute scores

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of detailed scores, or None if unavailable
        """
        try:
            _, detailed = self.aesthetic_assessor.assess_with_details(image_path)
            return detailed
        except Exception as e:
            print(f"Warning: Could not get detailed scores: {e}")
            return None

    def get_method_name(self) -> str:
        """Return method name for reporting."""
        return f"CLIP_Aesthetic_{self.model_name}_{self.aggregation_method}"

    def get_config(self) -> Dict[str, any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'aggregation_method': self.aggregation_method
        })
        
        # Add prompt information for clarity
        if hasattr(self, 'aesthetic_assessor'):
            prompt_summary = self.aesthetic_assessor.get_prompt_summary()
            config['prompts'] = {
                'contrastive_pairs': prompt_summary['contrastive_pairs'],
                'positive_attributes': prompt_summary['positive_attributes'],
                'negative_attributes': prompt_summary['negative_attributes'],
                'total_prompts': prompt_summary['total_prompts']
            }
            # Include actual prompt texts
            config['prompt_texts'] = {
                'contrastive_pairs': self.aesthetic_assessor.contrastive_prompts,
                'positive_attributes': self.aesthetic_assessor.positive_prompts,
                'negative_attributes': self.aesthetic_assessor.negative_prompts
            }
        
        return config

    def __repr__(self) -> str:
        return (
            f"CLIPAestheticAssessor("
            f"model_name='{self.model_name}', "
            f"pretrained='{self.pretrained}', "
            f"aggregation='{self.aggregation_method}')"
        )


class LearnedCLIPAestheticAssessor(CLIPAestheticAssessor):
    """
    CLIP-based aesthetic assessor using prompts learned from PhotoTriage dataset.

    This variant loads contrastive pairs from a YAML config file instead of
    using hardcoded prompts. The prompts are learned from user feedback in
    the PhotoTriage dataset by analyzing thousands of user-provided reasons
    for preferring one image over another.

    Example:
        >>> assessor = LearnedCLIPAestheticAssessor(
        >>>     prompts_file="configs/learned_aesthetic_prompts.yaml",
        >>>     model_name="ViT-B-32",
        >>>     device="cuda"
        >>> )
        >>> score = assessor.assess_image("photo.jpg")
    """

    def __init__(
        self,
        prompts_file: str = "configs/learned_aesthetic_prompts.yaml",
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        enable_cache: bool = True,
        aggregation_method: str = "weighted",
    ):
        """
        Initialize CLIP aesthetic assessor with learned prompts.

        Args:
            prompts_file: Path to YAML file with learned prompts
            model_name: OpenCLIP model architecture
            pretrained: Pretrained checkpoint
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache quality scores
            aggregation_method: How to combine scores
        """
        # Load prompts from YAML file
        prompts_path = Path(prompts_file)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        with open(prompts_path, 'r') as f:
            prompts_data = yaml.safe_load(f)

        # Extract contrastive pairs
        contrastive_pairs = prompts_data.get('contrastive_pairs', [])
        if not contrastive_pairs:
            raise ValueError(f"No contrastive_pairs found in {prompts_file}")

        # Convert to tuples
        contrastive_pairs = [tuple(pair) for pair in contrastive_pairs]

        # Override class-level prompts before calling parent __init__
        self.CONTRASTIVE_PAIRS = contrastive_pairs

        # For learned prompts, we don't use separate positive/negative attributes
        # All quality assessment comes from the contrastive pairs
        self.POSITIVE_ATTRIBUTES = []
        self.NEGATIVE_ATTRIBUTES = []

        # Store prompts file for reporting
        self.prompts_file = str(prompts_file)

        # Initialize parent class (which will use our overridden prompts)
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            enable_cache=enable_cache,
            aggregation_method=aggregation_method
        )

    def get_method_name(self) -> str:
        """Return method name for reporting."""
        return f"CLIP_Learned_{self.model_name}_{self.aggregation_method}"

    def get_config(self) -> Dict[str, any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'prompts_file': self.prompts_file,
            'num_contrastive_pairs': len(self.CONTRASTIVE_PAIRS)
        })
        return config

    def __repr__(self) -> str:
        return (
            f"LearnedCLIPAestheticAssessor("
            f"prompts_file='{self.prompts_file}', "
            f"model_name='{self.model_name}', "
            f"aggregation='{self.aggregation_method}')"
        )


# Backward compatibility: Export prompt definitions
__all__ = ['CLIPAestheticAssessor', 'LearnedCLIPAestheticAssessor']
