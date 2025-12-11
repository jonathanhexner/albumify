"""
Siamese ResNet50 for pairwise image quality ranking.

Ported from Series-Photo-Selection/models/ResNet50.py with improvements:
- Custom Bottleneck architecture
- Differential learning rates (10x for FC, 1x for backbone)
- Siamese architecture with difference layer before FC
- Trainable by default for end-to-end fine-tuning

Usage:
    model = SiameseResNet50()
    output = model(img1, img2)  # img1, img2: (batch, 3, 224, 224)
"""
import torch
import torch.nn as nn
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """ResNet Bottleneck block with 4x expansion."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # 1x1 conv (compression)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # 3x3 conv (bottleneck)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        # 1x1 conv (expansion)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SiameseResNet50(nn.Module):
    """
    Siamese ResNet50 for pairwise ranking.
    
    Architecture:
        Image1 → ResNet50 → feat1 ─┐
                                     ├→ diff = feat1 - feat2 → FC → output(2)
        Image2 → ResNet50 → feat2 ─┘
    
    The ResNet backbone is shared (Siamese) and the difference of features
    is computed before the final FC layer.
    """

    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize Siamese ResNet50.
        
        Args:
            num_classes: Number of output classes (default=2 for pairwise ranking)
            pretrained: Load ImageNet pre-trained weights
        """
        super(SiameseResNet50, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load ImageNet pre-trained weights (except FC layer)
        if pretrained:
            resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            pretrained_dict = resnet50.state_dict()
            # Exclude FC layer weights (we have different number of classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc')}
            self.load_state_dict(pretrained_dict, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a ResNet layer with multiple blocks."""
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            self.groups, self.base_width, self.dilation, norm_layer
        ))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        """
        Forward pass for Siamese ResNet.
        
        Args:
            x1: First image batch (batch_size, 3, 224, 224)
            x2: Second image batch (batch_size, 3, 224, 224)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Process image 1 through shared ResNet
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool(x1)
        x1 = x1.reshape(x1.size(0), -1)

        # Process image 2 through shared ResNet
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x2 = self.avgpool(x2)
        x2 = x2.reshape(x2.size(0), -1)

        # Compute difference before FC layer (key Siamese component)
        distance = x1 - x2
        out = self.fc(distance)

        return out

    def get_10x_lr_params(self):
        """Get parameters for 10x learning rate (FC layer)."""
        for m in self.named_modules():
            if 'fc' in m[0]:
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_1x_lr_params(self):
        """Get parameters for 1x learning rate (backbone)."""
        for m in self.named_modules():
            if 'fc' not in m[0]:
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def make_siamese_resnet50(num_classes=2, pretrained=True):
    """
    Factory function to create Siamese ResNet50.
    
    Args:
        num_classes: Number of output classes (default=2)
        pretrained: Load ImageNet pre-trained weights
        
    Returns:
        SiameseResNet50 model
    """
    return SiameseResNet50(num_classes=num_classes, pretrained=pretrained)

