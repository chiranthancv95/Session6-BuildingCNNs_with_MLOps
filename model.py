import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)  # 28x28x1 -> 28x28x12
        self.bn1 = nn.BatchNorm2d(12)
        
        # First Block
        self.conv2 = nn.Conv2d(12, 16, 3, padding=1)  # 28x28x12 -> 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 28x28x16 -> 28x28x16
        self.bn3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28x16 -> 14x14x16
        self.dropout1 = nn.Dropout(0.08)  # Slightly reduced dropout
        
        # Second Block
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16 -> 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16 -> 14x14x16
        self.bn5 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14x16 -> 7x7x16
        self.dropout2 = nn.Dropout(0.08)  # Slightly reduced dropout
        
        # Third Block
        self.conv6 = nn.Conv2d(16, 16, 3)  # 7x7x16 -> 5x5x16
        self.bn6 = nn.BatchNorm2d(16)
        
        # Output Block
        self.conv7 = nn.Conv2d(16, 10, 1)  # 5x5x16 -> 5x5x10
        self.gap = nn.AdaptiveAvgPool2d(1)  # 5x5x10 -> 1x1x10

    def forward(self, x):
        # Input block with residual
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block with residual
        identity1 = self.conv2(x)  # Shortcut connection
        x = F.relu(self.bn2(identity1))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity1)  # Skip connection
        x = self.dropout1(self.pool1(x))
        
        # Second block with residual
        identity2 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = F.relu(x + identity2)  # Skip connection
        x = self.dropout2(self.pool2(x))
        
        # Third block
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Output block
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)                   