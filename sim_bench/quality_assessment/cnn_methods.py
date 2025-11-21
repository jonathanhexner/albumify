"""
CNN-based image quality assessment (NIMA-style).
Uses pre-trained networks to predict aesthetic/technical quality.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.registry import register_method


class NIMAModel(nn.Module):
    """NIMA-style model predicting quality distribution."""
    
    def __init__(self, backbone: str = 'mobilenet_v2', num_classes: int = 10):
        """
        Initialize NIMA model.
        
        Args:
            backbone: Backbone architecture ('mobilenet_v2', 'resnet50', 'efficientnet_b0')
            num_classes: Number of quality levels (default 10 for 1-10 scale)
        """
        super().__init__()
        
        if backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.features = base_model
            
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.features = base_model
            
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.features = base_model
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Quality prediction head
        self.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """Forward pass."""
        features = self.features(x)
        quality_dist = self.classifier(features)
        return quality_dist


@register_method('nima')
@register_method('cnn')
class NIMAQuality(QualityAssessor):
    """NIMA-style CNN quality assessment."""

    def __init__(
        self,
        backbone: str = 'mobilenet_v2',
        weights_path: str = None,
        device: str = 'cpu',
        batch_size: int = 8
    ):
        """
        Initialize NIMA quality assessor.
        
        Args:
            backbone: CNN backbone ('mobilenet_v2', 'resnet50', 'efficientnet_b0')
            weights_path: Path to fine-tuned weights (optional)
            device: Device to run on
            batch_size: Batch size for inference
        """
        super().__init__(device)
        
        self.backbone = backbone
        self.batch_size = batch_size
        
        # Create model
        self.model = NIMAModel(backbone=backbone)
        
        # Load fine-tuned weights if provided
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @classmethod
    def is_available(cls) -> bool:
        """Check if PyTorch dependencies are available."""
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            return False

    @classmethod
    def from_config(cls, config: Dict) -> 'NIMAQuality':
        """
        Create NIMAQuality from config dict.

        Args:
            config: Configuration dictionary with keys:
                - backbone: CNN architecture (default: 'mobilenet_v2')
                - weights: Path to weights file (optional)
                - device: Device to run on (default: 'cpu')
                - batch_size: Batch size (default: 8)

        Returns:
            Configured NIMAQuality instance
        """
        return cls(
            backbone=config.get('backbone', config.get('model', 'mobilenet_v2')),
            weights_path=config.get('weights_path', config.get('weights')),
            device=config.get('device', 'cpu'),
            batch_size=config.get('batch_size', 8)
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
        
        # Compute mean score from distribution
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
                    # Use zero tensor as placeholder
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
    
    def get_quality_distribution(self, image_path: str) -> np.ndarray:
        """
        Get full quality distribution (probabilities for scores 1-10).
        
        Args:
            image_path: Path to image
            
        Returns:
            Array of 10 probabilities
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            quality_dist = self.model(img_tensor)
        
        return quality_dist.cpu().numpy()[0]
    
    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'backbone': self.backbone,
            'batch_size': self.batch_size
        })
        return config


def train_nima(
    model: NIMAModel,
    train_loader,
    val_loader,
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Train NIMA model on quality dataset.
    
    Args:
        model: NIMA model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # EMD loss (Earth Mover's Distance)
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
            optimizer.step()
            
            train_loss += loss.item()
        
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
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    return model





