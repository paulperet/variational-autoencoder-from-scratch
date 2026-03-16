import torch
import torchvision
import torch.nn as nn
import torch.functional as F
    
class ResNet18(nn.Module):
    """Encoder that follows the 18-layer ResNet architecture"""
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        # Activation function
        self.ReLU = nn.ReLU()

        # First block parameters; stride 2 is used to downsample instead of pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Second block parameters
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)

        # Third block parameters; stride 2 is used to downsample instead of pooling
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(num_features=128)

        self.proj3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)

        # Fourth block parameters; stride 2 is used to downsample instead of pooling
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(num_features=256)

        self.proj4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)

        # Fifth block parameters; stride 2 is used to downsample instead of pooling
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn5_1 = nn.BatchNorm2d(num_features=512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(num_features=512)

        self.proj5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)

        # Output logits
        self.average_pool = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Linear(in_features=512, out_features=num_classes)
    
    def forward(self, input_image):
        # First block, no residual connection
        block_1 = self.ReLU(self.bn1(self.conv1(input_image)))
        block_1 = self.max_pool(block_1)

        # Second block
        block_2 = self.ReLU(self.bn2_1(self.conv2_1(block_1)))
        block_2 = self.bn2_2(self.conv2_2(block_2))
        block_2 = self.ReLU(block_2 + block_1)                 # Residual connection; identity shortcut

        # Third block
        block_3 = self.ReLU(self.bn3_1(self.conv3_1(block_2)))
        block_3 = self.bn3_2(self.conv3_2(block_3))
        block_3 = self.ReLU(block_3 + self.proj3(block_2))    # Residual connection; projection shortcut

        # Fourth block
        block_4 = self.ReLU(self.bn4_1(self.conv4_1(block_3)))
        block_4 = self.bn4_2(self.conv4_2(block_4))
        block_4 = self.ReLU(block_4 + self.proj4(block_3))    # Residual connection; projection shortcut

        # Fifth block
        block_5 = self.ReLU(self.bn5_1(self.conv5_1(block_4)))
        block_5 = self.bn5_2(self.conv5_2(block_5))
        block_5 = self.ReLU(block_5 + self.proj5(block_4))     # Residual connection; projection shortcut

        # Global Average Pool; adapts to the last dim to allow any image input size
        output = self.average_pool(block_5)

        # Output logits
        output = self.linear(output.squeeze())

        return output