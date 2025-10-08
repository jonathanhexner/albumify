"""
CNN feature method using pre-trained deep learning models.
Uses configurable distance measures for comparison.
"""

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List
from sim_bench.feature_extraction.base import BaseMethod


class CNNFeatureMethod(BaseMethod):
    """CNN features with configurable distance measures."""
    
    def __init__(self, method_config):
        if not TORCH_AVAILABLE:
            raise ImportError(f"PyTorch is required for deep method but not available: {TORCH_ERROR}")
        super().__init__(method_config)
    
    def _build_model(self, name="resnet50"):
        """Build and return the CNN backbone model."""
        import torchvision.models as models
        if name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone = nn.Sequential(*list(m.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone
    
    def _transform(self):
        """Get image preprocessing transforms."""
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract deep CNN features from images."""
        backbone = self.method_config.get('backbone', 'resnet50')
        print(f"Extracting deep features ({backbone})...")
        
        device = torch.device("cpu")
        model = self._build_model(backbone).to(device)
        tr = self._transform()

        feats = []
        batch = []
        bs = int(self.method_config.get('batch_size', 16))
        for f in tqdm(image_paths, desc="deep: embeddings"):
            img = Image.open(f).convert('RGB')
            batch.append(tr(img))
            if len(batch) == bs:
                x = torch.stack(batch).to(device)
                y = model(x).view(x.size(0), -1)
                feats.append(y.cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            y = model(x).view(x.size(0), -1)
            feats.append(y.cpu().numpy())
        
        X = np.vstack(feats).astype('float32')
        if self.method_config.get('normalize', True):
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X
