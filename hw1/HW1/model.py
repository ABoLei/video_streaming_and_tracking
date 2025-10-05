import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution to reduce parameters"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=100, input_channels=3, base_channels=24):
        super().__init__()
        
        # Initial feature extraction with regular conv
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Efficient feature extraction blocks
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            SEBlock(base_channels * 2),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            SEBlock(base_channels * 4),
            nn.MaxPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 6, 3, 1, 1),
            nn.ReLU(inplace=True),
            SEBlock(base_channels * 6),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_channels * 6, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)      # 224x224 -> 56x56
        x = self.block1(x)    # 56x56 -> 28x28
        x = self.block2(x)    # 28x28 -> 14x14
        x = self.block3(x)    # 14x14 -> 1x1
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Alternative lightweight model for even fewer parameters
class LightweightModel(nn.Module):
    def __init__(self, num_classes=100, input_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Factory function to create models
def create_model(model_type='efficient', num_classes=100, input_channels=3):
    """
    Create a model based on type
    Args:
        model_type: 'efficient' for main model, 'lightweight' for minimal parameters
        num_classes: number of output classes
        input_channels: number of input channels (1 for grayscale, 3 for RGB)
    """
    if model_type == 'efficient':
        return ClassificationModel(num_classes, input_channels)
    elif model_type == 'lightweight':
        return LightweightModel(num_classes, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
