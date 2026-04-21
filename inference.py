import argparse
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
from models.autoencoder import AutoEncoder
from transforms import test_transforms, inverse_transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set device

device = "cpu"
pin_memory = True
use_amp = False
if torch.cuda.is_available():
    device = "cuda"
    use_amp=True

def inference(checkpoint_path, image_path):

    # Load the checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device)
    bottleneck_size = checkpoint["bottleneck_size"]
    

    # Create the model

    model = AutoEncoder(bottleneck=bottleneck_size).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Convert the image to 224x244
    image = Image.open(image_path)
    input = test_transforms(image).to(device)

    # Pass our image to the model
    output = model(input.unsqueeze(0)).squeeze()
    
    # Plot the input and reconstructed image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Inference')
    ax1.imshow(inverse_transforms(input).permute(1,2,0))
    ax1.set_title("Input Image")
    ax2.imshow(inverse_transforms(output).permute(1,2,0).detach().numpy())
    ax2.set_title("Reconstructed Image")

    plt.show()



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to an image",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Name of the model",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    inference(
        checkpoint_path=args.checkpoint,
        image_path=args.image_path,
    )