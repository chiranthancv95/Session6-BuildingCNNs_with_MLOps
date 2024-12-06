import torch
import torch.nn as nn
import torch.nn.functional as F

# class MNISTNet(nn.Module):
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         # Input Block
#         self.conv1 = nn.Conv2d(1, 16, 3)  # 28x28x1 -> 26x26x16
#         self.bn1 = nn.BatchNorm2d(16)
        
#         # First Block
#         self.conv2 = nn.Conv2d(16, 32, 3)  # 26x26x16 -> 24x24x32
#         self.bn2 = nn.BatchNorm2d(32)
#         self.dropout1 = nn.Dropout(0.01)
        
#         # Second Block
#         self.conv3 = nn.Conv2d(32, 16, 1)  # 24x24x32 -> 24x24x16
#         self.bn3 = nn.BatchNorm2d(16)
#         self.conv4 = nn.Conv2d(16, 16, 3)  # 24x24x16 -> 22x22x16
#         self.bn4 = nn.BatchNorm2d(16)
#         self.dropout2 = nn.Dropout(0.01)
        
#         # Third Block
#         self.conv5 = nn.Conv2d(16, 16, 3)  # 22x22x16 -> 20x20x16
#         self.bn5 = nn.BatchNorm2d(16)
#         self.conv6 = nn.Conv2d(16, 32, 3)  # 20x20x16 -> 18x18x32
#         self.bn6 = nn.BatchNorm2d(32)
#         self.dropout3 = nn.Dropout(0.01)
        
#         # Output Block
#         self.conv7 = nn.Conv2d(32, 10, 1)  # 18x18x32 -> 18x18x10
#         self.gap = nn.AdaptiveAvgPool2d(1)  # 18x18x10 -> 1x1x10

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout1(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.dropout2(x)
        
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = self.dropout3(x)
        
#         x = self.conv7(x)
#         x = self.gap(x)
#         x = x.view(-1, 10)
#         return F.log_softmax(x, dim=1)



class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv_block1=nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            # nn.Dropout(0.01)
                  ) #outputSize=26 from 28
    
        self.conv_block2=nn.Sequential(
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
            # nn.Dropout(0.01)
                  ) #outputSize= 24

        self.maxpool_block1=nn.Sequential(
            nn.MaxPool2d(2,2)
            ) #outputSize= 12

        self.conv_block3=nn.Sequential(
            nn.Conv2d(32,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            # nn.Dropout(0.01)
                  ) #outputSize= 10

        self.conv_block4=nn.Sequential(
            nn.Conv2d(16,10,3),
            nn.ReLU(),
            nn.BatchNorm2d(10)
            # nn.Dropout(0.01)
                  ) #outputSize= 8

        self.conv_block5=nn.Sequential(
            nn.Conv2d(10,10,3),
            nn.ReLU(),
            nn.BatchNorm2d(10)
            # nn.Dropout(0.01)
                  ) #outputSize= 6
    
        self.gap=nn.Sequential(
            nn.AvgPool2d(6,6)
        ) #outputSize= 1

        self.conv_block6=nn.Sequential(
            nn.Conv2d(10,10,1),
            # nn.ReLU(),
            # nn.BatchNorm2d(10)
            # nn.Dropout(0.01)
                  ) #outputSize= 1
    def forward(self, x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.maxpool_block1(x)
        x=self.conv_block3(x)
        x=self.conv_block4(x)
        x=self.conv_block5(x)
        x=self.gap(x)
        x=self.conv_block6(x)

        x=x.view(-1,10)
        return F.log_softmax(x, dim=-1)
