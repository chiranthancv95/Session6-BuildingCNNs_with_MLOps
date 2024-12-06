import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv_block1=nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
                  ) #outputSize=26 from 28

        self.conv_block2=nn.Sequential(
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
                  ) #outputSize= 24

        self.maxpool_block1=nn.Sequential(
            nn.MaxPool2d(2,2)
            ) #outputSize= 12

        self.conv_block3=nn.Sequential(
            nn.Conv2d(32,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
                  ) #outputSize= 10

        self.conv_block4=nn.Sequential(
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
                  ) #outputSize= 8

        self.conv_block5=nn.Sequential(
            nn.Conv2d(16,10,3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1)
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
