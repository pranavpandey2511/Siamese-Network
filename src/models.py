import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

# Import the loss function class
from contrastive_loss import ContrastiveLoss

class SiameseVanilla(nn.Module):
    def __init__(self):
        super(SiameseVanilla, self).__init__()
        self.Convolve = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.Linear = nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 24)
        )
    def forward(self, x_1, x_2):
        '''
        Keeping the passing of 2 inputs through the network explicit here for the sake of transperancy
        '''
        x_1 = self.Convolve(x_1)
        x_1 = x_1.reshape(x_1.size()[0], -1)
        x_1 = self.Linear(x_1)

        x_2 = self.Convolve(x_2)
        x_2 = x_2.reshape(x_2.size()[0], -1)
        x_2 = self.Linear(x_2)
        return x_1, x_2