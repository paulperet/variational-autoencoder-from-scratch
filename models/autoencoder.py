import torch
import torchvision
import torch.nn as nn
import torch.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def forward(self, x):
        return x
    
class Encoder(nn.Module):
    """Encoder that follows the 18-layer ResNet architecture"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=3)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=3)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=3)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.average_pool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ReLU = nn.ReLU()
        self.linear = nn.Linear(in_features=512*3*3, out_features=1000)
    
    def forward(self, x):
        x = self.ReLU(self.conv1(x));print(x.shape)
        x = self.max_pool(x);print(x.shape)
        x = self.ReLU(self.conv2_1(x));print(x.shape)
        x = self.ReLU(self.conv2_2(x));print(x.shape)
        x = self.ReLU(self.conv3_1(x));print(x.shape)
        x = self.ReLU(self.conv3_2(x));print(x.shape)
        x = self.ReLU(self.conv4_1(x));print(x.shape)
        x = self.ReLU(self.conv4_2(x));print(x.shape)
        x = self.ReLU(self.conv5_1(x));print(x.shape)
        x = self.ReLU(self.conv5_2(x));print(x.shape)
        return x

random_noise = torch.randn((1,3,224,224))

print(Encoder().forward(random_noise))