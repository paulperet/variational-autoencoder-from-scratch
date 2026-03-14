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

        # Activation function
        self.ReLU = nn.ReLU()

        # First block parameters; stride 2 is used to downsample instead of pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second block parameters
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)

        # Third block parameters; stride 2 is used to downsample instead of pooling
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=3)
        self.bn3_1 = nn.BatchNorm2d(num_features=128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn3_2 = nn.BatchNorm2d(num_features=128)

        # Fourth block parameters; stride 2 is used to downsample instead of pooling
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=3)
        self.bn4_1 = nn.BatchNorm2d(num_features=256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.bn4_2 = nn.BatchNorm2d(num_features=256)

        # Fifth block parameters; stride 2 is used to downsample instead of pooling
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=3)
        self.bn5_1 = nn.BatchNorm2d(num_features=512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.bn5_2 = nn.BatchNorm2d(num_features=512)

        self.average_pool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=512*3*3, out_features=1000)
    
    def forward(self, input_image):
        # First block, no residual connection
        block_1 = self.ReLU(self.bn1(self.conv1(input_image)))
        block_1 = self.max_pool(block_1)

        # Second block
        block_2 = self.ReLU(self.bn2_1(self.conv2_1(block_1)))
        block_2 = self.ReLU(self.bn2_2(self.conv2_2(block_2)))
        block_2 = block_2 #+ block_1 # Residual connection

        # Third block
        block_3 = self.ReLU(self.bn3_1(self.conv3_1(block_2)))
        block_3 = self.ReLU(self.bn3_2(self.conv3_2(block_3)))
        block_3 = block_3 #+ block_2 # Residual connection

        # Fourth block
        block_4 = self.ReLU(self.bn4_1(self.conv4_1(block_3)))
        block_4 = self.ReLU(self.bn4_2(self.conv4_2(block_4)))
        block_4 = block_4 #+ block_3 # Residual connection

        # Fifth block
        block_5 = self.ReLU(self.bn5_1(self.conv5_1(block_4)))
        block_5 = self.ReLU(self.bn5_2(self.conv5_2(block_5)))
        block_5 = block_5 #+ block_4 # Residual connection

        # Average pool
        output = self.average_pool(block_5)
        print(output.shape)

        # Flattening the input
        #output = self.flatten(output)

        return output

random_noise = torch.randn((1,3,224,224))
Encoder().forward(random_noise)