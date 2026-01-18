"""Siamese model wrapper for unified quality assessment interface."""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
from sim_bench.datasets.transform_factory import create_transform, FixScaleCrop

logger = logging.getLogger(__name__)


class SiameseQualityModel(BaseQualityModel):
    """Siamese E2E model for pairwise image quality comparison."""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cpu'):
        """
        Load Siamese model from checkpoint.
        
        Args:
            checkpoint_path: Path to best_model.pt checkpoint
            device: Device to run on
        """
        super().__init__(name='Siamese-E2E', device=device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = SiameseCNNRanker(self.config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create transform
        transform = create_transform(self.config)
        self.transform = transform if transform is not None else FixScaleCrop(crop_size=224)
        
        logger.info(f"Loaded Siamese model from epoch {checkpoint['epoch']}, "
                   f"val_acc={checkpoint['val_acc']:.3f}")
    
    def score_image(self, image_path: Path) -> float:
        """Siamese model only supports pairwise comparison."""
        raise NotImplementedError("Siamese model only supports pairwise comparison via compare_images()")
    
    def compare_images(self, image1_path: Path, image2_path: Path) -> Dict:
        """
        Native pairwise comparison.
        
        Returns prediction where 1 = img1 better, 0 = img2 better.
        """
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        img1_t = self.transform(img1).unsqueeze(0).to(self.device)
        img2_t = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img1_t, img2_t)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
        
        probs_np = probs[0].cpu().numpy()
        confidence = float(probs_np[pred])
        
        return {
            'prediction': pred,
            'confidence': confidence,
            'score_img1': None,  # Not applicable for Siamese
            'score_img2': None
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'SiameseQualityModel':
        """Create from config dict."""
        checkpoint_path = Path(config['checkpoint'])
        device = config.get('device', 'cpu')
        return cls(checkpoint_path, device)
