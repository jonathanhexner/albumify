"""AVA ResNet model wrapper for unified quality assessment interface."""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.models.ava_resnet import AVAResNet, create_transform

logger = logging.getLogger(__name__)


class AVAQualityModel(BaseQualityModel):
    """AVA ResNet model for aesthetic quality scoring (1-10 scale)."""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cpu'):
        """
        Load AVA model from checkpoint.
        
        Args:
            checkpoint_path: Path to best_model.pt checkpoint
            device: Device to run on
        """
        super().__init__(name='AVA-ResNet', device=device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = AVAResNet(self.config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Get output mode
        self.output_mode = self.config['model'].get('output_mode', 'distribution')
        
        # Create transform
        transform_config = self.config.get('transform', {})
        self.transform = create_transform(transform_config, is_train=False)
        
        val_score = checkpoint.get('val_spearman', 'N/A')
        logger.info(f"Loaded AVA model from epoch {checkpoint['epoch']}, "
                   f"val_spearman={val_score}, mode={self.output_mode}")
    
    def score_image(self, image_path: Path) -> float:
        """
        Score image aesthetics on 1-10 scale.
        
        Returns:
            Aesthetic score (1-10, higher = better)
        """
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_t)
            score = self._compute_mean_score(output)
        
        return score
    
    def _compute_mean_score(self, output: torch.Tensor) -> float:
        """Compute mean score from model output."""
        if self.output_mode == 'distribution':
            probs = F.softmax(output, dim=1)
            scores = torch.arange(1, 11, dtype=torch.float32, device=self.device)
            mean_score = (probs * scores).sum(dim=1).item()
        else:  # regression
            mean_score = output.squeeze(-1).clamp(1, 10).item()
        
        return mean_score
    
    @classmethod
    def from_config(cls, config: Dict) -> 'AVAQualityModel':
        """Create from config dict."""
        checkpoint_path = Path(config['checkpoint'])
        device = config.get('device', 'cpu')
        return cls(checkpoint_path, device)
