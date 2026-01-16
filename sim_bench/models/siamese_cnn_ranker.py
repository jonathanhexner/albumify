"""
Simple Siamese CNN + MLP for pairwise image ranking.

Architecture:
    Image1 → CNN → feat1 ─┐
                           ├→ diff → MLP → output(2)
    Image2 → CNN → feat2 ─┘

Supports:
- VGG16 and ResNet50 backbones
- End-to-end training (CNN + MLP)
- Paper-style preprocessing (aspect ratio + padding)
- Flexible MLP architecture
"""
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from typing import List, Tuple, Optional


class SiameseCNNRanker(nn.Module):
    """
    Siamese CNN + MLP for pairwise ranking.
    
    Simple, focused architecture for end-to-end training of CNN backbones.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Siamese CNN + MLP ranker from config dict.
        
        Args:
            config: Configuration dict with keys:
                - cnn_backbone: 'vgg16' or 'resnet50'
                - mlp_hidden_dims: List of hidden layer dimensions
                - dropout: Dropout rate (default: 0.0)
                - activation: 'relu' or 'tanh' (default: 'relu')
                - pretrained: Load ImageNet weights (default: True)
                - use_paper_preprocessing: Aspect ratio + padding (default: False)
                - padding_mean_color: RGB mean for padding (default: ImageNet mean)
        """
        super().__init__()
        
        # Extract config with defaults
        cnn_backbone = config.get('cnn_backbone', 'resnet50')
        mlp_hidden_dims = config.get('mlp_hidden_dims', [128, 64])
        dropout = config.get('dropout', 0.0)
        activation = config.get('activation', 'relu')
        pretrained = config.get('pretrained', True)
        use_paper_preprocessing = config.get('use_paper_preprocessing', False)
        padding_mean_color = config.get('padding_mean_color')
        if padding_mean_color:
            padding_mean_color = tuple(padding_mean_color)
        else:
            padding_mean_color = (0.485, 0.456, 0.406)
        
        self.cnn_backbone_name = cnn_backbone
        self.use_paper_preprocessing = use_paper_preprocessing
        self.padding_mean_color = padding_mean_color or (0.485, 0.456, 0.406)
        
        # Create CNN backbone
        if cnn_backbone == 'vgg16':
            if pretrained:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            else:
                vgg = models.vgg16(weights=None)
            # VGG16 classifier outputs 4096-dim features
            self.backbone = nn.Sequential(*list(vgg.features), vgg.avgpool, nn.Flatten())
            self.feature_dim = 25088  # 512 * 7 * 7 after avgpool
            # Add VGG's first FC layer to get to 4096
            self.backbone.add_module('fc6', vgg.classifier[0])  # Linear(25088, 4096)
            self.backbone.add_module('relu6', nn.ReLU(inplace=True))
            self.backbone.add_module('dropout6', nn.Dropout(p=0.5))
            self.backbone.add_module('fc7', vgg.classifier[3])  # Linear(4096, 4096)
            self.backbone.add_module('relu7', nn.ReLU(inplace=True))
            self.backbone.add_module('dropout7', nn.Dropout(p=0.5))
            self.feature_dim = 4096
            
        elif cnn_backbone == 'resnet50':
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet50(weights=None)
            # ResNet50 without final FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.feature_dim = 2048
            
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # Build MLP
        activation_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        
        layers = []
        in_dim = self.feature_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                activation_fn,
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final output layer (2-way classification)
        layers.append(nn.Linear(in_dim, 2))

        self.mlp = nn.Sequential(*layers)

        # Initialize MLP weights (match reference implementation)
        self._initialize_mlp_weights()

        # Create preprocessing transform
        self.preprocess = self._create_transform()

    def _initialize_mlp_weights(self):
        """Initialize MLP weights to match reference implementation."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # Kaiming normal initialization (same as reference)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_transform(self):
        """Create image preprocessing transform."""
        if self.use_paper_preprocessing:
            # Paper-style: aspect-ratio preserving resize + padding
            if self.cnn_backbone_name == 'vgg16':
                target_size = 224
            else:  # resnet50
                target_size = 224
            
            return AspectRatioResizeAndPad(
                target_size=target_size,
                mean_color=self.padding_mean_color,
                normalize_mean=[0.485, 0.456, 0.406],
                normalize_std=[0.229, 0.224, 0.225]
            )
        else:
            # Standard ImageNet preprocessing
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, return_feats: bool = False):
        """
        Forward pass for Siamese network.

        Args:
            img1: First image batch (batch_size, 3, H, W)
            img2: Second image batch (batch_size, 3, H, W)
            return_feats: If True, return (log_probs, feat1, feat2, diff) for diagnostics

        Returns:
            If return_feats=False:
                Logits (batch_size, 2)
                Index 0: score for img2 > img1
                Index 1: score for img1 > img2
            If return_feats=True:
                Tuple of (logits, feat1, feat2, diff)
        """
        # Extract features through shared CNN
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Compute difference
        diff = feat1 - feat2

        # Pass through MLP
        logits = self.mlp(diff)

        if return_feats:
            return logits, feat1, feat2, diff
        return logits
    
    def get_trainable_params(self):
        """Get all trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def get_1x_lr_params(self):
        """Get CNN backbone parameters (for lower learning rate)."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param
    
    def get_10x_lr_params(self):
        """Get MLP head parameters (for higher learning rate)."""
        for param in self.mlp.parameters():
            if param.requires_grad:
                yield param
    
    def get_differential_lr_groups(self, base_lr: float = 1e-3, fc_multiplier: float = 10.0):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            base_lr: Base learning rate for backbone
            fc_multiplier: Multiplier for MLP learning rate
            
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr},
            {'params': self.mlp.parameters(), 'lr': base_lr * fc_multiplier}
        ]


class AspectRatioResizeAndPad:
    """
    Resize image preserving aspect ratio, then pad to square.
    
    Matches PhotoTriage paper preprocessing.
    """
    
    def __init__(
        self,
        target_size: int = 224,
        mean_color: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.mean_color = mean_color
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    
    def __call__(self, img):
        """Apply transform to PIL image."""
        # Get original size
        w, h = img.size
        
        # Resize so larger dimension equals target_size
        if w > h:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        else:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)
        
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)
        
        # Pad to square with mean color
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        
        # Create padding tensor with mean color
        padded = torch.ones(3, self.target_size, self.target_size)
        for c in range(3):
            padded[c] = self.mean_color[c]
        
        # Place resized image in center
        top = pad_h // 2
        left = pad_w // 2
        padded[:, top:top+new_h, left:left+new_w] = img_tensor
        
        # Normalize
        return self.normalize(padded)

