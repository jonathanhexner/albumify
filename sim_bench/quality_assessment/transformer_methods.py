"""
Transformer-based image quality assessment using Vision Transformers (ViT).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, List

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.registry import register_method

try:
    from transformers import ViTModel, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ViTQualityModel(nn.Module):
    """Vision Transformer for quality assessment."""
    
    def __init__(
        self, 
        model_name: str = 'google/vit-base-patch16-224',
        num_quality_levels: int = 10,
        freeze_backbone: bool = False
    ):
        """
        Initialize ViT quality model.
        
        Args:
            model_name: Pre-trained ViT model name
            num_quality_levels: Number of quality levels to predict
            freeze_backbone: Whether to freeze ViT backbone
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for ViT. "
                "Install with: pip install transformers"
            )
        
        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Quality prediction head
        hidden_size = self.vit.config.hidden_size
        self.quality_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_quality_levels),
            nn.Softmax(dim=1)
        )
        
    def forward(self, pixel_values):
        """Forward pass."""
        # Get ViT features
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Predict quality distribution
        quality_dist = self.quality_head(cls_output)
        
        return quality_dist


@register_method('vit')
@register_method('transformer')
class ViTQuality(QualityAssessor):
    """Vision Transformer quality assessment."""

    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        weights_path: str = None,
        device: str = 'cpu',
        batch_size: int = 4
    ):
        """
        Initialize ViT quality assessor.
        
        Args:
            model_name: Pre-trained ViT model name
                       Options: 'google/vit-base-patch16-224' (default, 86M params)
                               'google/vit-large-patch16-224' (307M params)
                               'google/vit-base-patch32-224' (88M params, faster)
            weights_path: Path to fine-tuned weights (optional)
            device: Device to run on
            batch_size: Batch size for inference
        """
        super().__init__(device)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for ViT. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Create model
        print(f"Loading ViT model: {model_name}...")
        self.model = ViTQualityModel(model_name=model_name)
        
        # Load fine-tuned weights if provided
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing (ViT expects 224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # ViT uses 0.5 normalization
                std=[0.5, 0.5, 0.5]
            )
        ])

    @classmethod
    def is_available(cls) -> bool:
        """Check if transformers dependencies are available."""
        try:
            import torch
            from transformers import ViTModel
            return True
        except ImportError:
            return False

    @classmethod
    def from_config(cls, config: Dict) -> 'ViTQuality':
        """
        Create ViTQuality from config dict.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name or checkpoint (default: 'google/vit-base-patch16-224')
                - weights: Path to weights file (optional)
                - device: Device to run on (default: 'cpu')
                - batch_size: Batch size (default: 4)

        Returns:
            Configured ViTQuality instance
        """
        return cls(
            model_name=config.get('model', config.get('model_name', config.get('checkpoint', 'google/vit-base-patch16-224'))),
            weights_path=config.get('weights_path', config.get('weights')),
            device=config.get('device', 'cpu'),
            batch_size=config.get('batch_size', 4)
        )

    def assess_image(self, image_path: str) -> float:
        """
        Assess image quality.
        
        Args:
            image_path: Path to image
            
        Returns:
            Quality score (mean of predicted distribution)
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            quality_dist = self.model(img_tensor)
        
        # Compute mean score from distribution (1-10 scale)
        scores = torch.arange(1, 11, dtype=torch.float32).to(self.device)
        mean_score = torch.sum(quality_dist * scores, dim=1)
        
        return float(mean_score.item())
    
    def assess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Assess quality of multiple images (efficient batched version).
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Array of quality scores
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load and transform images
            batch_tensors = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                quality_dists = self.model(batch)
            
            # Compute mean scores
            scores = torch.arange(1, 11, dtype=torch.float32).to(self.device)
            mean_scores = torch.sum(quality_dists * scores, dim=1)
            
            all_scores.extend(mean_scores.cpu().numpy())
        
        return np.array(all_scores)
    
    def get_attention_maps(self, image_path: str) -> np.ndarray:
        """
        Get attention maps from ViT to visualize which regions affect quality.
        
        Args:
            image_path: Path to image
            
        Returns:
            Attention maps from last layer [num_heads, num_patches, num_patches]
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.vit(
                pixel_values=img_tensor,
                output_attentions=True
            )
        
        # Get attention from last layer
        attentions = outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
        
        return attentions[0].cpu().numpy()
    
    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'batch_size': self.batch_size
        })
        return config


def train_vit_quality(
    model: ViTQualityModel,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 5e-5,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Train ViT quality model.
    
    Args:
        model: ViT quality model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate (smaller for transformers)
        device: Device to train on
        save_path: Path to save best model
        
    Returns:
        Trained model
    """
    model = model.to(device)
    
    # Use AdamW optimizer (better for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # EMD loss
    def emd_loss(pred_dist, target_dist):
        """Earth Mover's Distance between distributions."""
        cdf_pred = torch.cumsum(pred_dist, dim=1)
        cdf_target = torch.cumsum(target_dist, dim=1)
        return torch.mean(torch.abs(cdf_pred - cdf_target))
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = emd_loss(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = emd_loss(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    return model





