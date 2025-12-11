"""
Multi-feature pairwise ranker for PhotoTriage.

Combines CLIP embeddings + CNN mid-level features + IQA features
to learn image quality ranking with margin ranking loss.

Architecture:
    Image → [CLIP(frozen), ResNet-layer3(frozen), IQA] → Concat → [LayerNorm (optional)] → MLP → Scalar Score

For pairs: Score(img1) vs Score(img2) with margin ranking loss.
For series: Score all images, rank by scores.

This module uses composition pattern to cleanly handle different feature combinations
without if-statement hell.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# MultiFeatureConfig class removed - now using dict-based config directly from YAML


# ============================================================================
# Feature Extractor Components (Composition Pattern)
# ============================================================================

class FeatureExtractorComponent(ABC):
    """
    Base class for feature extractor components.

    Each component extracts a specific type of feature (CLIP, CNN, IQA) and
    knows its own dimension. This enables clean composition without if-statements.
    """

    @abstractmethod
    def extract(self, image_path: str, image_pil: Optional[Image.Image] = None) -> torch.Tensor:
        """
        Extract features from an image.

        Args:
            image_path: Path to the image file
            image_pil: Optional pre-loaded PIL Image (for efficiency)

        Returns:
            Feature tensor of shape (dim,)
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the feature dimension."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the component name."""
        pass


class CLIPExtractorComponent(FeatureExtractorComponent):
    """Extract CLIP semantic features."""

    def __init__(self, config: dict):
        """
        Initialize CLIP feature extractor.

        Args:
            config: Configuration dict with CLIP model settings
        """
        from sim_bench.vision_language.clip import CLIPModel

        self.config = config
        clip_model = config.get('clip_model', 'ViT-B-32')
        clip_checkpoint = config.get('clip_checkpoint', 'laion2b_s34b_b79k')
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading CLIP: {clip_model} ({clip_checkpoint})")
        self.model = CLIPModel(
            model_name=clip_model,
            pretrained=clip_checkpoint,
            device=device,
            enable_cache=False  # Handle caching at dataset level
        )
        self._dim = self.model.get_embedding_dim()
        logger.info(f"CLIP feature dimension: {self._dim}")

    @torch.no_grad()
    def extract(self, image_path: str, image_pil: Optional[Image.Image] = None) -> torch.Tensor:
        """Extract CLIP features (L2 normalized)."""
        # Use path-based encoding since CLIPModel.encode_images expects paths
        # and returns numpy arrays
        embedding = self.model.encode_images([image_path])[0]  # numpy array (clip_dim,)
        return torch.from_numpy(embedding).float()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "clip"


class CNNExtractorComponent(FeatureExtractorComponent):
    """Extract CNN features using ResNet or VGG16."""

    def __init__(self, config: dict):
        """
        Initialize CNN feature extractor.

        Args:
            config: Configuration dict with CNN backbone and layer settings
        """
        self.config = config
        cnn_backbone = config.get('cnn_backbone', 'resnet50')
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        if cnn_backbone == "vgg16":
            from sim_bench.feature_extraction.vgg_features import VGGFeatureExtractor

            # Convert list to tuple if provided
            padding_mean_color = config.get('padding_mean_color')
            mean_color = tuple(padding_mean_color) if padding_mean_color else None
            cnn_freeze_mode = config.get('cnn_freeze_mode', 'all')
            use_paper_preprocessing = config.get('use_paper_preprocessing', False)

            logger.info(f"Loading VGG16 (freeze_mode={cnn_freeze_mode}, paper_preproc={use_paper_preprocessing})")
            self.model = VGGFeatureExtractor(
                pretrained=True,
                freeze_until=cnn_freeze_mode,
                device=device,
                use_paper_preprocessing=use_paper_preprocessing,
                mean_color=mean_color
            )
        else:
            from sim_bench.feature_extraction.resnet_features import ResNetFeatureExtractor

            cnn_layer = config.get('cnn_layer', 'layer4')
            logger.info(f"Loading CNN: {cnn_backbone} (layer: {cnn_layer})")
            self.model = ResNetFeatureExtractor(
                backbone=cnn_backbone,
                layer=cnn_layer,
                device=device,
                pretrained=True
            )

        self._dim = self.model.output_dim
        logger.info(f"CNN feature dimension: {self._dim}")

    def extract(self, image_path: str, image_pil: Optional[Image.Image] = None) -> torch.Tensor:
        """
        Extract CNN features.

        Note: No @torch.no_grad() decorator here - features may need gradients
        if CNN is being fine-tuned (cnn_freeze_mode != "all").
        """
        if image_pil is None:
            image_pil = Image.open(image_path).convert('RGB')

        # When CNN is trainable, gradients will flow through
        # When CNN is frozen, no gradients regardless
        features = self.model.extract(image_pil)  # (cnn_dim,)
        return features

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "cnn"


class IQAExtractorComponent(FeatureExtractorComponent):
    """Extract IQA (Image Quality Assessment) features using rule-based methods."""

    def __init__(self, config: dict):
        """
        Initialize IQA feature extractor.

        Args:
            config: Configuration dict with IQA feature settings
        """
        from sim_bench.quality_assessment.rule_based import RuleBasedQuality

        self.config = config

        # Determine which IQA features to use
        self.iqa_features = []
        if config.get('use_sharpness', True):
            self.iqa_features.append('sharpness')
        if config.get('use_exposure', True):
            self.iqa_features.append('exposure')
        if config.get('use_colorfulness', True):
            self.iqa_features.append('colorfulness')
        if config.get('use_contrast', True):
            self.iqa_features.append('contrast')

        self._dim = len(self.iqa_features)

        # Initialize RuleBasedQuality assessor
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.assessor = RuleBasedQuality(device=device)

        logger.info(f"IQA features ({self._dim}): {self.iqa_features}")

    def extract(self, image_path: str, image_pil: Optional[Image.Image] = None) -> torch.Tensor:
        """
        Extract IQA features using RuleBasedQuality assessor.

        Args:
            image_path: Path to image file (required for IQA)
            image_pil: Unused (IQA loads image itself)

        Returns:
            Tensor of shape (iqa_dim,) with normalized IQA features
        """
        # Get all IQA metrics at once
        detailed = self.assessor.get_detailed_scores(image_path)

        # Extract requested features in order
        iqa_values = []
        for feature_name in self.iqa_features:
            if feature_name == 'sharpness':
                iqa_values.append(detailed['sharpness_normalized'])
            elif feature_name == 'exposure':
                iqa_values.append(detailed['exposure'])
            elif feature_name == 'colorfulness':
                iqa_values.append(detailed['colorfulness_normalized'])
            elif feature_name == 'contrast':
                iqa_values.append(detailed['contrast'])

        return torch.tensor(iqa_values, dtype=torch.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "iqa"


# ============================================================================
# Multi-Feature Extractor (Composition)
# ============================================================================

class MultiFeatureExtractor(nn.Module):
    """
    Extracts multi-level features from images using composition pattern.

    This class composes different feature extractors (CLIP, CNN, IQA) based on
    configuration, eliminating if-statement hell and making the code extensible.

    Attributes:
        config: Configuration object
        extractors: List of active feature extractor components
        total_dim: Total dimension of concatenated features
    """

    def __init__(self, config: dict):
        """
        Initialize multi-feature extractor with composition pattern.

        Args:
            config: Configuration dict specifying which features to use
        """
        super().__init__()
        self.config = config

        # Compose feature extractors based on config
        self.extractors: List[FeatureExtractorComponent] = []

        if config.get('use_clip', True):
            self.extractors.append(CLIPExtractorComponent(config))

        if config.get('use_cnn_features', True):
            self.extractors.append(CNNExtractorComponent(config))

        if config.get('use_iqa_features', True):
            self.extractors.append(IQAExtractorComponent(config))

        # Validate at least one feature type is enabled
        if len(self.extractors) == 0:
            raise ValueError("No features enabled! Set at least one of use_clip, use_cnn_features, or use_iqa_features to True.")

        # Calculate total dimension (composition makes this trivial!)
        self.total_dim = sum(ext.dim for ext in self.extractors)

        # Log configuration
        feature_summary = ", ".join([f"{ext.name}({ext.dim})" for ext in self.extractors])
        logger.info(f"MultiFeatureExtractor initialized: {feature_summary} → total_dim={self.total_dim}")

    def extract_all(self, image_path: str) -> torch.Tensor:
        """
        Extract all enabled features for an image (raw, unnormalized).

        Returns raw features for caching. Normalization is applied later
        in the model's encode() method to keep cached features general.

        Args:
            image_path: Path to image file

        Returns:
            Concatenated feature vector of shape (total_dim,)
            Features are in order: [clip, cnn, iqa] (only enabled ones)
        """
        # Load image once for efficiency
        image_pil = Image.open(image_path).convert('RGB')

        # Extract from all components (raw, no normalization)
        features = [
            extractor.extract(image_path, image_pil)
            for extractor in self.extractors
        ]

        # Concatenate raw features
        return torch.cat(features)

    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get dimensions of all active features.

        Returns:
            Dictionary mapping feature names to dimensions
        """
        return {ext.name: ext.dim for ext in self.extractors}


# ============================================================================
# Pairwise Ranker Model (True Siamese Network with Late Fusion)
# ============================================================================

class MultiFeaturePairwiseRanker(nn.Module):
    """
    True Siamese network for pairwise ranking with late fusion architecture.

    Architecture (when all features enabled):
        Image1 → [CLIP; CNN; IQA] (raw from cache)
                      ↓
              Split & L2-normalize each group
                      ↓
        [CLIP_norm; CNN_norm] → Visual Tower → visual_emb (64-dim)
                                                    ↓
                                    [visual_emb; IQA_norm] → Siamese Tower → emb1 ─┐
                                                                                    ├→ Comparison → P(img1>img2)
        Image2 → ... (same path, shared weights) ────────────────────────────→ emb2 ─┘
    
    Comparison modes:
        - "diff_only": emb1-emb2 → MLP (simpler, stronger ranking bias)
        - "full": [emb1, emb2, emb1-emb2] → MLP (more expressive)

    Late Fusion Benefits:
        - Visual features (CLIP+CNN: 1536-dim) compressed before IQA joins
        - IQA (4-dim) meaningful relative to compressed visual (64-dim)
        - No dropout on IQA features (only 4 dims, can't afford to drop)

    Handles all feature combinations:
        - Only IQA: skip visual tower, IQA goes directly to siamese tower
        - Only visual (CLIP/CNN): skip IQA fusion
        - Mixed: full late fusion architecture

    Attributes:
        config: Configuration object
        feature_extractor: Multi-feature extractor (frozen)
        clip_dim, cnn_dim, iqa_dim: Dimensions of each feature group
        visual_tower: Processes CLIP+CNN before IQA fusion (optional)
        siamese_tower: Processes combined features
        comparison_head: Compares embeddings from both images
    """

    def __init__(self, config: dict):
        """
        Initialize Siamese network with late fusion.

        Args:
            config: Configuration dictionary (from YAML)
        """
        super().__init__()
        self.config = config

        # Feature extractor (frozen) - tells us what features are enabled
        self.feature_extractor = MultiFeatureExtractor(config)
        
        # Get dimensions from feature extractor
        feature_dims = self.feature_extractor.get_feature_dims()
        self.clip_dim = feature_dims.get('clip', 0)
        self.cnn_dim = feature_dims.get('cnn', 0)
        self.iqa_dim = feature_dims.get('iqa', 0)
        
        # Track what we have
        self.visual_dim = self.clip_dim + self.cnn_dim
        self.has_visual = self.visual_dim > 0
        self.has_iqa = self.iqa_dim > 0
        
        logger.info(f"Feature groups: CLIP={self.clip_dim}, CNN={self.cnn_dim}, IQA={self.iqa_dim}")
        logger.info(f"Has visual: {self.has_visual}, Has IQA: {self.has_iqa}")

        # Choose activation function
        activation = config.get('activation', 'relu')
        if activation == "tanh":
            self.activation_fn = nn.Tanh()
            activation_str = "tanh"
        else:  # default "relu"
            self.activation_fn = nn.ReLU()
            activation_str = "relu"

        # Visual Tower: processes CLIP + CNN (if any visual features enabled)
        # Skip if use_visual_tower=False (for paper replication with raw features)
        use_visual_tower = config.get('use_visual_tower', True)
        if self.has_visual and use_visual_tower:
            visual_hidden_dim = config.get('visual_hidden_dim', 256)
            visual_embedding_dim = config.get('visual_embedding_dim', 64)
            visual_dropout = config.get('visual_dropout', 0.3)
            self.visual_tower = nn.Sequential(
                nn.Linear(self.visual_dim, visual_hidden_dim),
                self.activation_fn,
                nn.Dropout(visual_dropout),
                nn.Linear(visual_hidden_dim, visual_embedding_dim)
            )
            self.visual_out_dim = visual_embedding_dim
            logger.info(f"Visual tower ({activation_str}): {self.visual_dim} → {visual_hidden_dim} → {visual_embedding_dim}")
        else:
            self.visual_tower = None
            # If no visual tower but have visual features, use them directly
            self.visual_out_dim = self.visual_dim if (self.has_visual and not use_visual_tower) else 0

        # Siamese Tower input: visual_emb (if any) + IQA (if any)
        siamese_input_dim = self.visual_out_dim + self.iqa_dim
        
        # Build Siamese Tower
        mlp_hidden_dims = config.get('mlp_hidden_dims', [128, 64])
        dropout = config.get('dropout', 0.3)
        tower_layers = []
        in_dim = siamese_input_dim
        for hidden_dim in mlp_hidden_dims:
            tower_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh() if activation == "tanh" else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.siamese_tower = nn.Sequential(*tower_layers)
        self.embedding_dim = in_dim
        logger.info(f"Siamese tower ({activation_str}): {siamese_input_dim} → {' → '.join(map(str, mlp_hidden_dims))}")

        # Comparison head: takes embeddings and outputs P(img1 > img2)
        # diff_only: just emb1-emb2 (simpler, stronger inductive bias)
        # full: [emb1, emb2, emb1-emb2] (more expressive)
        comparison_mode = config.get('comparison_mode', 'diff_only')
        self.comparison_mode = comparison_mode
        if comparison_mode == "diff_only":
            comparison_input_dim = self.embedding_dim
        else:  # "full"
            comparison_input_dim = 3 * self.embedding_dim
        
        # Paper uses 2-way Softmax (outputs probabilities for both classes)
        # We use LogSoftmax for numerical stability with NLLLoss
        comparison_hidden_dim = config.get('comparison_hidden_dim', 64)
        comparison_dropout = config.get('comparison_dropout', 0.3)
        self.comparison_head = nn.Sequential(
            nn.Linear(comparison_input_dim, comparison_hidden_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU(),
            nn.Dropout(comparison_dropout),
            nn.Linear(comparison_hidden_dim, 2),  # 2 outputs for 2-way classification
            nn.LogSoftmax(dim=-1)  # Log probabilities for numerical stability
        )
        logger.info(f"Comparison head ({activation_str}, {comparison_mode}): {comparison_input_dim} → {comparison_hidden_dim} → 2 (2-way softmax)")

        # Log total parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")

    def _split_features(self, features: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Split concatenated features into separate groups.
        
        Features come from cache in order: [clip, cnn, iqa]
        
        Args:
            features: Concatenated features (batch_size, total_dim)
            
        Returns:
            Tuple of (clip_feat, cnn_feat, iqa_feat), each None if not present
        """
        offset = 0
        
        clip_feat = None
        if self.clip_dim > 0:
            clip_feat = features[..., offset:offset + self.clip_dim]
            offset += self.clip_dim
            
        cnn_feat = None
        if self.cnn_dim > 0:
            cnn_feat = features[..., offset:offset + self.cnn_dim]
            offset += self.cnn_dim
            
        iqa_feat = None
        if self.iqa_dim > 0:
            iqa_feat = features[..., offset:offset + self.iqa_dim]
            
        return clip_feat, cnn_feat, iqa_feat

    def _get_architecture_summary(self) -> str:
        """Get human-readable architecture summary."""
        parts = []
        if self.has_visual:
            parts.append(f"Visual({self.visual_dim}→{self.visual_out_dim})")
        if self.has_iqa:
            parts.append(f"IQA({self.iqa_dim})")
        
        siamese_in = self.visual_out_dim + self.iqa_dim
        mlp_hidden_dims = self.config.get('mlp_hidden_dims', [128, 64])
        siamese_dims = [siamese_in] + list(mlp_hidden_dims)
        
        cmp_in = self.embedding_dim if self.comparison_mode == "diff_only" else 3 * self.embedding_dim
        comparison_hidden_dim = self.config.get('comparison_hidden_dim', 64)
        return (f"[{' + '.join(parts)}] → Siamese({' → '.join(map(str, siamese_dims))}) → "
                f"Compare[{self.comparison_mode}]({cmp_in} → {comparison_hidden_dim} → 1)")

    def get_architecture_metadata(self) -> Dict:
        """Get complete architecture metadata for logging/saving."""
        return {
            'architecture_type': 'late_fusion_siamese',
            'clip_dim': self.clip_dim,
            'cnn_dim': self.cnn_dim,
            'iqa_dim': self.iqa_dim,
            'visual_dim': self.visual_dim,
            'has_visual': self.has_visual,
            'has_iqa': self.has_iqa,
            'visual_hidden_dim': self.config.get('visual_hidden_dim', 256) if self.has_visual else 0,
            'visual_embedding_dim': self.visual_out_dim,
            'siamese_input_dim': self.visual_out_dim + self.iqa_dim,
            'mlp_hidden_dims': self.config.get('mlp_hidden_dims', [128, 64]),
            'embedding_dim': self.embedding_dim,
            'comparison_mode': self.comparison_mode,
            'comparison_input_dim': self.embedding_dim if self.comparison_mode == "diff_only" else 3 * self.embedding_dim,
            'comparison_hidden_dim': self.config.get('comparison_hidden_dim', 64),
            'total_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture_summary': self._get_architecture_summary(),
            'visual_dropout': self.config.get('visual_dropout', 0.3) if self.has_visual else 0,
            'siamese_dropout': self.config.get('dropout', 0.3),
            'comparison_dropout': self.config.get('comparison_dropout', 0.3),
        }

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through the network to get embeddings.
        
        Applies:
        1. Split features into groups
        2. L2-normalize each group
        3. Process visual features through visual tower (if any)
        4. Late fusion: combine visual_emb + IQA
        5. Process through siamese tower

        Args:
            features: Pre-extracted features (batch_size, total_dim)

        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        # Split into groups
        clip_feat, cnn_feat, iqa_feat = self._split_features(features)
        
        parts_for_siamese = []
        
        # Process visual features (CLIP + CNN) if present
        use_feature_normalization = self.config.get('use_feature_normalization', False)
        if self.has_visual:
            visual_parts = []
            if clip_feat is not None:
                # Optionally L2 normalize CLIP (disabled by default for paper replication)
                if use_feature_normalization:
                    visual_parts.append(F.normalize(clip_feat, p=2, dim=-1))
                else:
                    visual_parts.append(clip_feat)
            if cnn_feat is not None:
                # Optionally L2 normalize CNN (disabled by default for paper replication)
                if use_feature_normalization:
                    visual_parts.append(F.normalize(cnn_feat, p=2, dim=-1))
                else:
                    visual_parts.append(cnn_feat)

            # Concatenate visual features
            visual_concat = torch.cat(visual_parts, dim=-1)

            # Process through visual tower (if enabled) or use raw features
            if self.visual_tower is not None:
                visual_emb = self.visual_tower(visual_concat)
                parts_for_siamese.append(visual_emb)
            else:
                # No visual tower: use raw visual features directly (paper replication mode)
                parts_for_siamese.append(visual_concat)
        
        # Add IQA features (late fusion) - optionally normalize, no dropout
        if self.has_iqa and iqa_feat is not None:
            if self.config.use_feature_normalization:
                iqa_norm = F.normalize(iqa_feat, p=2, dim=-1)
                parts_for_siamese.append(iqa_norm)
            else:
                parts_for_siamese.append(iqa_feat)
        
        # Combine and pass through siamese tower
        combined = torch.cat(parts_for_siamese, dim=-1)
        embedding = self.siamese_tower(combined)
        
        return embedding

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compare two images through Siamese network.

        Args:
            feat1: Features for image 1 (batch_size, total_dim)
            feat2: Features for image 2 (batch_size, total_dim)

        Returns:
            Log probabilities for 2-way classification, shape (batch_size, 2)
            - Index 0: log P(img2 > img1)
            - Index 1: log P(img1 > img2)
        """
        # Encode both through shared network
        emb1 = self.encode(feat1)
        emb2 = self.encode(feat2)

        # Combine based on comparison mode
        if self.comparison_mode == "diff_only":
            combined = emb1 - emb2
        else:  # "full"
            combined = torch.cat([emb1, emb2, emb1 - emb2], dim=-1)

        # Comparison head outputs (batch_size, 2) log probabilities
        log_probs = self.comparison_head(combined)
        return log_probs

    def forward_images(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass for VGG16 training.
        
        Args:
            img1, img2: Preprocessed images (batch_size, 3, H, W)
            
        Returns:
            Log probabilities (batch_size, 2)
        """
        # Find CNN model
        cnn_model = None
        for extractor in self.feature_extractor.extractors:
            if extractor.name == 'cnn':
                cnn_model = extractor.model
                break
        
        if cnn_model is None:
            raise ValueError("forward_images requires CNN features to be enabled")
        
        # Forward through CNN then MLP
        feat1 = cnn_model(img1)
        feat2 = cnn_model(img2)
        return self.forward(feat1, feat2)

    def get_cnn_transform(self):
        """Get CNN preprocessing transform for end-to-end training."""
        for extractor in self.feature_extractor.extractors:
            if extractor.name == 'cnn' and hasattr(extractor.model, 'preprocess'):
                return extractor.model.preprocess
        return None


# ============================================================================
# Dataset
# ============================================================================

class MultiFeaturePairwiseDataset(Dataset):
    """
    Dataset for pairwise ranking training.

    Each sample is a pair of images with a label indicating which is better.
    Returns pre-extracted features (not raw images) for efficiency.
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        image_dir: str,
        feature_extractor: MultiFeatureExtractor,
        feature_cache: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize pairwise dataset.

        Args:
            pairs_df: DataFrame with columns [image1, image2, winner]
                     winner is 0 if image1 is better, 1 if image2 is better
            image_dir: Directory containing images
            feature_extractor: MultiFeatureExtractor instance
            feature_cache: Optional dict mapping image_name → features
        """
        self.pairs_df = pairs_df
        self.image_dir = Path(image_dir)
        self.feature_extractor = feature_extractor
        self.feature_cache = feature_cache or {}

        logger.info(f"Dataset size: {len(pairs_df)} pairs")
        if self.feature_cache:
            logger.info(f"Feature cache size: {len(self.feature_cache)} images")

    def __len__(self):
        return len(self.pairs_df)

    def _get_features(self, image_name: str) -> torch.Tensor:
        """Get features for an image (from cache or extract)."""
        # Try cache with image_name as key (new format)
        if image_name in self.feature_cache:
            return self.feature_cache[image_name]

        # Fallback: try full path as key (old format for backwards compatibility)
        image_path = str(self.image_dir / image_name)
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]

        # Extract on-the-fly (slower but always works)
        features = self.feature_extractor.extract_all(image_path)
        return features

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]

        # Get features for both images
        feat1 = self._get_features(row['image1'])
        feat2 = self._get_features(row['image2'])

        # Winner label: 0 = image1 is better, 1 = image2 is better
        winner = int(row['winner'])

        return {
            'feat1': feat1,
            'feat2': feat2,
            'winner': torch.tensor(winner, dtype=torch.long),
            'image1': row['image1'],
            'image2': row['image2']
        }


# EndToEndPairDataset removed - now defined in training scripts where it's actually used


def compute_pairwise_accuracy(log_probs: torch.Tensor, winners: torch.Tensor) -> float:
    """
    Compute pairwise accuracy: how often does the model predict the correct winner?

    Args:
        log_probs: Log probabilities from model (batch_size, 2)
                   - Index 0: log P(img2 > img1)
                   - Index 1: log P(img1 > img2)
        winners: Ground truth winners (batch_size,), 0 if img1 wins, 1 if img2 wins

    Returns:
        Accuracy as float [0, 1]
    """
    # Predicted winner: argmax of log probabilities
    # If log_probs[:, 1] > log_probs[:, 0], then pred_winner = 1 (img1 wins)
    # If log_probs[:, 0] > log_probs[:, 1], then pred_winner = 0 (img2 wins)
    # But our encoding is: winner=0 means img1 wins, winner=1 means img2 wins
    # So we need to invert: if model says img1 wins (argmax=1), winner should be 0
    pred_winners = log_probs.argmax(dim=-1)  # 1 if img1 wins, 0 if img2 wins
    pred_winners = 1 - pred_winners  # Invert to match winner encoding

    # Compare with ground truth
    correct = (pred_winners == winners).sum().item()
    total = len(winners)

    return correct / total
