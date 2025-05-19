import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=50, input_channels=1):
        super().__init__()
        self.in_channels = 64

        # Adapt input conv for input_channels (e.g., 1 for spectrogram)
        self.conv1 = nn.Conv2d(input_channels, self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 layers configuration: 2 blocks per layer
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)      # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64, H/4, W/4]

        x = self.layer1(x)     # [B, 64, H/4, W/4]
        x = self.layer2(x)     # [B, 128, H/8, W/8]
        x = self.layer3(x)     # [B, 256, H/16, W/16]
        x = self.layer4(x)     # [B, 512, H/32, W/32]

        x = self.avgpool(x)    # [B, 512, 1, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)         # [B, num_classes]

        return x


"""
class Bottleneck(nn.Module):
    expansion = 4  # output channels are 4x the "base" channels
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        mid_channels = out_channels
        
        # 1x1 conv reduces channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 3x3 conv does spatial filtering
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 1x1 conv expands channels
        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # shortcut to match dimensions if needed
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

class ResNet50(nn.Module):
    def __init__(self, num_classes=50, input_channels=1):
        super().__init__()
        self.in_channels = 64
        
        # Change input_channels from 3 to 1 (or as passed)
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # The rest remains unchanged...
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)  # Drop 50% of activations
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        layers = []
        
        # If input and output dimensions differ, or stride != 1, downsample shortcut
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        
        # First block in layer with possible downsampling
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        
        # Remaining blocks, stride=1
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # -> [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # -> [B, 64, H/4, W/4]
        
        x = self.layer1(x)  # -> [B, 256, H/4, W/4]
        x = self.layer2(x)  # -> [B, 512, H/8, W/8]
        x = self.layer3(x)  # -> [B, 1024, H/16, W/16]
        x = self.layer4(x)  # -> [B, 2048, H/32, W/32]
        
        x = self.avgpool(x) # -> [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # -> [B, 2048]
        x = self.dropout(x)      # Apply dropout before classification
        x = self.fc(x)           # -> [B, num_classes]
        
        return x
"""