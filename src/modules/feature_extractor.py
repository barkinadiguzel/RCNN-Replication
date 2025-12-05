import torch
import torch.nn as nn
from layers.conv_layer import ConvLayer  

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvLayer(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(3, stride=2),
            ConvLayer(64, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            ConvLayer(192, 384, kernel_size=3, stride=1, padding=1),
            ConvLayer(384, 256, kernel_size=3, stride=1, padding=1),
            ConvLayer(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2)
        )

    def forward(self, x):
        return self.features(x)
