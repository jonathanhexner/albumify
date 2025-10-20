"""
ResNet-50 feature attribution using Grad-CAM.

Provides spatial attribution for deep learning features extracted with ResNet-50.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

from .base import BaseAttributionExtractor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.models as models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResNet50AttributionExtractor(BaseAttributionExtractor):
    """
    Extract features and compute Grad-CAM attributions for ResNet-50.
    
    Grad-CAM shows which spatial regions of the image contribute most
    to specific feature dimensions in the final feature vector.
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize ResNet-50 attribution extractor.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResNet-50 attribution")
        
        self.device = torch.device(device)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Storage for intermediate activations and gradients
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for layer4 (last conv layer before avgpool)
        self._register_hooks()
        
        # Image preprocessing transforms
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def _register_hooks(self):
        """Register forward and backward hooks for layer4."""
        def forward_hook(module, input, output):
            self.activations['layer4'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['layer4'] = grad_output[0].detach()
        
        self.model.layer4.register_forward_hook(forward_hook)
        self.model.layer4.register_full_backward_hook(backward_hook)
    
    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image_array)
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img_array
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract final feature vector (same as used in sim-bench).
        
        Args:
            image_path: Path to image
            
        Returns:
            Feature vector of shape (2048,)
        """
        img_tensor, _ = self._load_image(image_path)
        
        with torch.no_grad():
            # Forward through all layers except final FC
            x = self.model.conv1(img_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            features = x.view(x.size(0), -1)
        
        # Normalize (same as sim-bench)
        features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-12)
        
        return features.cpu().numpy()[0]
    
    def compute_attribution(
        self,
        image_path: str,
        feature_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM attribution map.
        
        Args:
            image_path: Path to image
            feature_indices: Which feature dimensions to visualize (if None, uses all)
            
        Returns:
            Tuple of (attribution_map, original_image)
            attribution_map: [H, W] array with values in [0, 1]
            original_image: [H, W, 3] RGB image array
        """
        img_tensor, img_array = self._load_image(image_path)
        img_tensor.requires_grad = True
        
        # Forward pass
        x = self.model.conv1(img_tensor)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # Activation stored via hook
        x = self.model.avgpool(x)
        features = x.view(x.size(0), -1)
        
        # Normalize
        features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-12)
        
        # Backward pass for selected features
        if feature_indices is None:
            target = features.sum()
        else:
            target = features[0, feature_indices].sum()
        
        self.model.zero_grad()
        target.backward()
        
        # Get activations and gradients
        activations = self.activations['layer4']  # (1, 2048, 7, 7)
        gradients = self.gradients['layer4']      # (1, 2048, 7, 7)
        
        # Compute weights (global average of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, img_array
    
    def analyze_feature_importance(
        self,
        image_path: str,
        top_k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze which feature dimensions are most active for an image.
        
        Args:
            image_path: Path to image
            top_k: Number of top features to return
            
        Returns:
            Dictionary with:
                - 'features': Full feature vector
                - 'top_indices': Indices of top-k features
                - 'top_values': Values of top-k features
        """
        features = self.extract_features(image_path)
        
        # Find top-k most active features
        top_indices = np.argsort(np.abs(features))[-top_k:][::-1]
        top_values = features[top_indices]
        
        return {
            'features': features,
            'top_indices': top_indices,
            'top_values': top_values
        }

