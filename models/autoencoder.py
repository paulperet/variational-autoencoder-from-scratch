import torch
import torchvision
import torch.nn as nn
import torch.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, bottleneck=1024):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(bottleneck=bottleneck)
        self.decoder = Decoder(bottleneck=bottleneck)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Encoder(nn.Module):
    """Encoder that follows the 18-layer ResNet architecture"""
    def __init__(self, bottleneck):
        super(Encoder, self).__init__()

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
        self.linear = nn.Linear(in_features=512, out_features=bottleneck)

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

        # Global Average Pool; works only if current input size is 7x7 meaning that input image is 224x224
        output = block_5.view(-1,512*7*7) # self.average_pool(block_5)

        # Output logits
        output = self.linear(output.squeeze())

        return output
    
class Decoder(nn.Module):
    """Decoder that mimic the 18-layer ResNet architecture in reverse"""
    def __init__(self, bottleneck):
        super(Decoder, self).__init__()

        # Activation function
        self.ReLU = nn.ReLU()

        self.linear = nn.Linear(in_features=bottleneck, out_features=512*7*7)

        # First block parameters; stride 2 is used to downsample instead of pooling
        self.deconv1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(num_features=512)
        self.deconv1_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1_2 = nn.BatchNorm2d(num_features=512)

        self.proj1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=1, stride=2, output_padding=1)
        
        # Second block parameters; stride 2 is used to downsample instead of pooling
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=256)
        self.deconv2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=256)

        self.proj2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=1, stride=2, output_padding=1)

        # Third block parameters; stride 2 is used to downsample instead of pooling
        self.deconv3_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=128)
        self.deconv3_2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.bn3_2 = nn.BatchNorm2d(num_features=128)

        self.proj3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1, stride=2, output_padding=1)

        # Fourth block parameters
        self.deconv4_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=64)
        self.deconv4_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4_2 = nn.BatchNorm2d(num_features=64)

        self.proj4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=1, stride=2, output_padding=1)

        # Fifth block parameters; stride 2 is used to downsample instead of pooling
        self.deconv5_1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, bottleneck):
        # First block, no residual connection

        bottleneck_nn = self.linear(bottleneck)
        bottleneck_nn = bottleneck_nn.view(-1,512,7,7)

        block_1 = self.ReLU(self.bn1_1(self.deconv1_1(bottleneck_nn)))
        block_1 = self.bn1_2(self.deconv1_2(block_1))
        block_1 = self.ReLU(block_1 + self.proj1(bottleneck_nn))  # Residual connection; projection shortcut

        # Second block
        block_2 = self.ReLU(self.bn2_1(self.deconv2_1(block_1)))
        block_2 = self.bn2_2(self.deconv2_2(block_2))
        block_2 = self.ReLU(block_2 + self.proj2(block_1))        # Residual connection; projection shortcut

        # Third block
        block_3 = self.ReLU(self.bn3_1(self.deconv3_1(block_2)))
        block_3 = self.bn3_2(self.deconv3_2(block_3))
        block_3 = self.ReLU(block_3 + self.proj3(block_2))    # Residual connection; projection shortcut

        # Fourth block
        block_4 = self.ReLU(self.bn4_1(self.deconv4_1(block_3)))
        block_4 = self.bn4_2(self.deconv4_2(block_4))
        block_4 = self.ReLU(block_4 + self.proj4(block_3))    # Residual connection; identity shortcut

        # Fifth block
        block_5 = self.ReLU(self.deconv5_1(block_4)) # Last block; no residual connection

        return block_5

#random_noise = torch.randn((32,3,224,224))
#Encoder(bottleneck=256).forward(random_noise)

#AutoEncoder().forward(random_noise)