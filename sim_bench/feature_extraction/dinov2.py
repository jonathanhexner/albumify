"""
DINOv2 feature extraction using Meta's self-supervised vision transformer.
Supports multiple model variants (small, base, large, giant).
"""

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms as T
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List
from sim_bench.feature_extraction.base import BaseMethod


class DINOv2Method(BaseMethod):
    """DINOv2 vision transformer features with configurable model sizes."""
    
    def __init__(self, method_config):
        if not TORCH_AVAILABLE:
            raise ImportError(f"PyTorch is required for DINOv2 but not available: {TORCH_ERROR}")
        super().__init__(method_config)
        self.model = None
    
    def _build_model(self, variant="base"):
        """
        Build and return DINOv2 model.
        
        Args:
            variant: Model size - 'small', 'base', 'large', or 'giant'
        """
        variant = variant.lower()
        valid_variants = ["small", "base", "large", "giant"]
        
        if variant not in valid_variants:
            raise ValueError(f"Invalid DINOv2 variant: {variant}. Choose from {valid_variants}")
        
        model_name = f"dinov2_vit{variant[0]}14"  # dinov2_vits14, dinov2_vitb14, etc.
        
        try:
            # Load from torch.hub
            model = torch.hub.load('facebookresearch/dinov2', model_name)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 model '{model_name}': {e}")
    
    def _get_transform(self):
        """Get image preprocessing transforms for DINOv2."""
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract DINOv2 features from images."""
        import torch
        
        variant = self.method_config.get('variant', 'base')
        batch_size = int(self.method_config.get('batch_size', 16))
        device_name = self.method_config.get('device', 'cpu')
        
        print(f"Extracting DINOv2-{variant} features from {len(image_paths)} images (batch_size={batch_size})...")
        
        # Build model once
        if self.model is None:
            self.model = self._build_model(variant)
        
        device = torch.device(device_name)
        self.model = self.model.to(device)
        transform = self._get_transform()
        
        features = []
        batch = []
        
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=f"DINOv2-{variant}", unit="img"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch.append(transform(img))
                    
                    if len(batch) == batch_size:
                        x = torch.stack(batch).to(device)
                        # DINOv2 outputs features directly
                        y = self.model(x)
                        features.append(y.cpu().numpy())
                        batch = []
                except Exception as e:
                    print(f"\nWarning: Failed to process {img_path}: {e}")
                    # Add zero vector for failed images to maintain alignment
                    if features:
                        features.append(np.zeros((1, features[0].shape[1]), dtype='float32'))
                    continue
            
            # Process remaining batch
            if batch:
                x = torch.stack(batch).to(device)
                y = self.model(x)
                features.append(y.cpu().numpy())
        
        X = np.vstack(features).astype('float32')
        
        # Optional L2 normalization
        if self.method_config.get('normalize', True):
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        return X

