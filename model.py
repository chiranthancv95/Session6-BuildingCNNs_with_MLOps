import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 28x28x1 -> 28x28x16
        self.bn1 = nn.BatchNorm2d(16)
        
        # First Block
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 28x28x16 -> 28x28x32
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)  # 28x28x32 -> 28x28x32
        self.bn3 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28x32 -> 14x14x32
        self.dropout1 = nn.Dropout(0.075)
        
        # Second Block with Skip Connection
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)  # 14x14x32 -> 14x14x32
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)  # 14x14x32 -> 14x14x32
        self.bn5 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14x32 -> 7x7x32
        self.dropout2 = nn.Dropout(0.075)
        
        # Output Block
        self.conv6 = nn.Conv2d(32, 16, 1)  # 7x7x32 -> 7x7x16
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 10, 1)  # 7x7x16 -> 7x7x10
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7x10 -> 1x1x10

    def forward(self, x):
        # Input block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(self.pool1(x))
        
        # Second block with skip connection
        identity = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = F.relu(x + identity)  # Skip connection
        x = self.dropout2(self.pool2(x))
        
        # Output block
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)                   