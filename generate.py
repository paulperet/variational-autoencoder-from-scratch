import argparse
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
from models.autoencoder import AutoEncoder
from models.variational_autoencoder import VariationalAutoEncoder
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

def generate_image_unguided(checkpoint_path, model_type="ae"):
    """Generate an image from a random point in the feature space"""

    # Load the checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device)
    bottleneck_size = checkpoint["bottleneck_size"]
    

    # Create the model

    if model_type == "vae":
        model = VariationalAutoEncoder(bottleneck=bottleneck_size).to(device=device)
    else:
        model = AutoEncoder(bottleneck=bottleneck_size).to(device=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])

    # Sample a random vector in the latent space

    random_vector = torch.randn(1, bottleneck_size)

    # Pass our vector to the model
    output = model.decode(random_vector).squeeze()
    
    # Plot the input and reconstructed image
    plt.title('Model Inference, random point in the feature space')
    plt.imshow(inverse_transforms(output).permute(1,2,0).detach().cpu().numpy())
    plt.show()

def generate_image_guided(checkpoint_path, image_path, model_type="ae"):
    """Generate an image starting from a encoded image"""
    # Load the checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device)
    bottleneck_size = checkpoint["bottleneck_size"]

    # Create the model

    if model_type == "vae":
        model = VariationalAutoEncoder(bottleneck=bottleneck_size).to(device=device)
    else:
        model = AutoEncoder(bottleneck=bottleneck_size).to(device=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])

    # Convert the image to 224x244
    image = Image.open(image_path)
    input = test_transforms(image).to(device)

    # Get the features
    features = model.encode(input.unsqueeze(0))
    print(features.max(), features.min())

    # Operations on the features: slightly move the point to get interesting results
    features += torch.randn(bottleneck_size) * 5

    # Pass our vector to the model
    output_sample_close_points = model.decode(features).squeeze()

    # Get the default reconstruction
    output_default = model(input.unsqueeze(0)).squeeze()
    
    # Plot the input and reconstructed image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Model Inference')
    ax1.imshow(inverse_transforms(input.cpu()).permute(1,2,0))
    ax1.set_title("Input Image")
    ax2.imshow(inverse_transforms(output_default).permute(1,2,0).detach().cpu().numpy())
    ax2.set_title("Reconstructed Image")
    ax3.imshow(inverse_transforms(output_sample_close_points).permute(1,2,0).detach().cpu().numpy())
    ax3.set_title("Generated image from a close point in the feature space")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-path",
        type=Path,
        required=False,
        help="Path to an image",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Name of the model",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model: ae, vae or b-vae (default: ae)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.image_path:
        generate_image_guided(
            checkpoint_path=args.checkpoint,
            image_path=args.image_path,
            model_type=args.model,
        )
    else:
        generate_image_unguided(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
        )
