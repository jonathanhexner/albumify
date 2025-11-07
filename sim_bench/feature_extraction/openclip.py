"""
OpenCLIP feature extraction using open-source CLIP models.
Supports various model architectures and pre-trained checkpoints.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List
from sim_bench.feature_extraction.base import BaseMethod


class OpenCLIPMethod(BaseMethod):
    """OpenCLIP vision-language model for image features."""
    
    def __init__(self, method_config):
        if not TORCH_AVAILABLE:
            raise ImportError(f"PyTorch is required for OpenCLIP but not available: {TORCH_ERROR}")
        if not OPENCLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required for OpenCLIP method. "
                "Install with: pip install open-clip-torch"
            )
        super().__init__(method_config)
        self.model = None
        self.preprocess = None
    
    def _build_model(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        """
        Build and return OpenCLIP model.
        
        Args:
            model_name: Model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained checkpoint (e.g., 'laion2b_s34b_b79k', 'openai')
        """
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model, preprocess
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenCLIP model '{model_name}' with pretrained='{pretrained}': {e}\n"
                f"Available models: {open_clip.list_pretrained()}"
            )
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract OpenCLIP image features."""
        import torch
        
        model_name = self.method_config.get('model', 'ViT-B-32')
        pretrained = self.method_config.get('pretrained', 'laion2b_s34b_b79k')
        batch_size = int(self.method_config.get('batch_size', 16))
        device_name = self.method_config.get('device', 'cpu')
        
        print(f"Extracting OpenCLIP features (model={model_name}, pretrained={pretrained})...")
        print(f"Processing {len(image_paths)} images (batch_size={batch_size})...")
        
        # Build model once
        if self.model is None:
            self.model, self.preprocess = self._build_model(model_name, pretrained)
        
        device = torch.device(device_name)
        self.model = self.model.to(device)
        
        features = []
        batch = []
        
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=f"OpenCLIP-{model_name}", unit="img"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch.append(self.preprocess(img))
                    
                    if len(batch) == batch_size:
                        x = torch.stack(batch).to(device)
                        # Extract image features using CLIP's image encoder
                        image_features = self.model.encode_image(x)
                        features.append(image_features.cpu().numpy())
                        batch = []
                except Exception as e:
                    print(f"\nWarning: Failed to process {img_path}: {e}")
                    # Add zero vector for failed images
                    if features:
                        features.append(np.zeros((1, features[0].shape[1]), dtype='float32'))
                    continue
            
            # Process remaining batch
            if batch:
                x = torch.stack(batch).to(device)
                image_features = self.model.encode_image(x)
                features.append(image_features.cpu().numpy())
        
        X = np.vstack(features).astype('float32')
        
        # Optional L2 normalization (recommended for CLIP)
        if self.method_config.get('normalize', True):
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        return X

