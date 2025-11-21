"""
MUSIQ (Multi-Scale Image Quality Transformer) implementation.
Uses pyiqa library for pre-trained weights, or custom architecture with weights loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math
import warnings

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.registry import register_method

# Try to import pyiqa for pre-trained MUSIQ weights
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False


class LearnablePositionalEncoding(nn.Module):
    """Learnable 2D positional encoding for variable-size images."""
    
    def __init__(self, embed_dim: int, max_size: int = 384):
        """
        Initialize learnable positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_size: Maximum image size (height or width)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_size = max_size
        
        # Learnable position embeddings for row and column
        self.row_embed = nn.Embedding(max_size, embed_dim)
        self.col_embed = nn.Embedding(max_size, embed_dim)
    
    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Generate positional encodings for image of size (h, w).
        
        Args:
            h: Image height
            w: Image width
            
        Returns:
            Positional encodings [h*w, embed_dim]
        """
        device = self.row_embed.weight.device
        
        # Create position indices
        row_pos = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).flatten()
        col_pos = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1).flatten()
        
        # Get embeddings
        row_emb = self.row_embed(row_pos)
        col_emb = self.col_embed(col_pos)
        
        # Combine row and column embeddings
        pos_emb = row_emb + col_emb
        
        return pos_emb


class MultiScaleTransformerBlock(nn.Module):
    """Transformer block for multi-scale processing."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class MUSIQModel(nn.Module):
    """MUSIQ Multi-Scale Image Quality Transformer."""
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 12,
        num_quality_levels: int = 10,
        patch_size: int = 16,
        max_size: int = 384
    ):
        """
        Initialize MUSIQ model.
        
        Args:
            embed_dim: Embedding dimension (default: 384)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of transformer layers (default: 12)
            num_quality_levels: Number of quality levels to predict (default: 10)
            patch_size: Patch size for image tokenization (default: 16)
            max_size: Maximum image size for positional encoding (default: 384)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_size = max_size
        
        # Patch embedding (conv layer to convert patches to embeddings)
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_size)
        
        # CLS token for global quality representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MultiScaleTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Quality prediction head
        self.quality_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_quality_levels),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W] (variable size)
            
        Returns:
            Quality distribution [B, num_quality_levels]
        """
        B, C, H, W = x.shape
        
        # Extract patches and embed
        # x: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.patch_embed(x)
        
        # Flatten spatial dimensions
        # x: [B, embed_dim, h, w] -> [B, h*w, embed_dim]
        B, embed_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, h*w, embed_dim]
        
        # Add positional encoding
        pos_emb = self.pos_encoding(h, w)  # [h*w, embed_dim]
        x = x + pos_emb.unsqueeze(0)  # [B, h*w, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+h*w, embed_dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract CLS token
        x = self.norm(x)
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # Predict quality distribution
        quality_dist = self.quality_head(cls_output)  # [B, num_quality_levels]
        
        return quality_dist


@register_method('musiq')
class MUSIQQuality(QualityAssessor):
    """MUSIQ Multi-Scale Image Quality Transformer with pre-trained weights."""
    
    def __init__(
        self,
        use_pyiqa: bool = True,
        weights_path: Optional[str] = None,
        device: str = 'cpu',
        batch_size: int = 1,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 12,
        patch_size: int = 16,
        max_size: int = 384
    ):
        """
        Initialize MUSIQ quality assessor.
        
        Args:
            use_pyiqa: Use pyiqa library for pre-trained weights (default: True)
            weights_path: Path to custom pre-trained weights (optional, overrides use_pyiqa)
            device: Device to run on
            batch_size: Batch size (default: 1, MUSIQ handles variable sizes)
            embed_dim: Embedding dimension (only used if not using pyiqa)
            num_heads: Number of attention heads (only used if not using pyiqa)
            num_layers: Number of transformer layers (only used if not using pyiqa)
            patch_size: Patch size for tokenization (only used if not using pyiqa)
            max_size: Maximum image size (only used if not using pyiqa)
        """
        super().__init__(device)
        
        self.use_pyiqa = use_pyiqa and PYIQA_AVAILABLE and weights_path is None
        self.batch_size = batch_size
        
        if self.use_pyiqa:
            # Use pyiqa library with pre-trained weights
            print(f"Loading MUSIQ from pyiqa library (pre-trained weights)...")
            try:
                # pyiqa handles image resizing automatically, so we don't need max_size
                self.model = pyiqa.create_metric('musiq', device=device)
                print("MUSIQ pre-trained model loaded successfully from pyiqa")
                print("Note: pyiqa automatically handles image resizing to prevent memory issues")
            except Exception as e:
                warnings.warn(
                    f"Failed to load MUSIQ from pyiqa: {e}. "
                    f"Falling back to custom implementation. "
                    f"Install pyiqa for pre-trained weights: pip install pyiqa"
                )
                self.use_pyiqa = False
        
        if not self.use_pyiqa:
            # Use custom implementation
            print(f"Initializing custom MUSIQ model (embed_dim={embed_dim}, layers={num_layers})...")
            if weights_path is None:
                warnings.warn(
                    "MUSIQ is being initialized without pre-trained weights. "
                    "Performance will be poor. "
                    "Install pyiqa (pip install pyiqa) or provide weights_path for pre-trained weights."
                )
            
            self.model = MUSIQModel(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                patch_size=patch_size,
                max_size=max_size
            )
            
            # Load pre-trained weights if provided
            if weights_path is not None:
                print(f"Loading MUSIQ weights from {weights_path}...")
                state_dict = torch.load(weights_path, map_location=device)
                self.model.load_state_dict(state_dict, strict=False)
                print("Custom MUSIQ weights loaded successfully")
            
            self.model = self.model.to(device)
            self.model.eval()
        
        # Image preprocessing (normalize only, no resizing)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if dependencies are available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MUSIQQuality':
        """
        Create MUSIQQuality from config dict.
        
        Args:
            config: Configuration dictionary with keys:
                - use_pyiqa: Use pyiqa library (default: True)
                - weights_path: Path to weights file (optional)
                - device: Device to run on (default: 'cpu')
                - batch_size: Batch size (default: 1)
                - embed_dim: Embedding dimension (only if not using pyiqa, default: 384)
                - num_heads: Number of attention heads (only if not using pyiqa, default: 8)
                - num_layers: Number of transformer layers (only if not using pyiqa, default: 12)
                - patch_size: Patch size (only if not using pyiqa, default: 16)
                - max_size: Maximum image size (only if not using pyiqa, default: 384)
        
        Returns:
            Configured MUSIQQuality instance
        """
        return cls(
            use_pyiqa=config.get('use_pyiqa', True),
            weights_path=config.get('weights_path', config.get('weights')),
            device=config.get('device', 'cpu'),
            batch_size=config.get('batch_size', 1),
            embed_dim=config.get('embed_dim', 384),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 12),
            patch_size=config.get('patch_size', 16),
            max_size=config.get('max_size', 384)
        )
    
    def _prepare_image(self, image_path: str) -> torch.Tensor:
        """
        Prepare image for MUSIQ with size limiting to prevent memory issues.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize to max_size while preserving aspect ratio to prevent memory issues
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # [C, H, W], [0, 1]
        
        # Normalize
        img_tensor = self.normalize(img_tensor)
        
        return img_tensor
    
    def assess_image(self, image_path: str) -> float:
        """
        Assess image quality.
        
        Args:
            image_path: Path to image
            
        Returns:
            Quality score (mean of predicted distribution or direct score)
        """
        if self.use_pyiqa:
            # pyiqa expects PIL Image or tensor
            img = Image.open(image_path).convert('RGB')
            
            # pyiqa's create_metric returns a callable that takes PIL Image or tensor
            with torch.no_grad():
                score = self.model(img)
            
            # pyiqa returns a score directly (typically 0-100 or 0-10 scale)
            # Convert to tensor if needed
            if isinstance(score, torch.Tensor):
                score = score.item()
            return float(score)
        else:
            # Custom implementation
            img_tensor = self._prepare_image(image_path).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                quality_dist = self.model(img_tensor)
            
            # Compute mean score from distribution (1-10 scale)
            scores = torch.arange(1, 11, dtype=torch.float32).to(self.device)
            mean_score = torch.sum(quality_dist * scores, dim=1)
            
            return float(mean_score.item())
    
    def assess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Assess quality of multiple images.
        
        Note: MUSIQ processes images at native resolution, so batching
        requires padding or processing individually. We process individually
        to maintain accuracy.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Array of quality scores
        """
        all_scores = []
        
        for img_path in image_paths:
            try:
                score = self.assess_image(img_path)
                all_scores.append(score)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                all_scores.append(5.0)  # Default middle score
        
        return np.array(all_scores)
    
    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'use_pyiqa': self.use_pyiqa,
            'batch_size': self.batch_size
        })
        if not self.use_pyiqa:
            config.update({
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'patch_size': self.patch_size,
                'max_size': self.max_size
            })
        return config
    

