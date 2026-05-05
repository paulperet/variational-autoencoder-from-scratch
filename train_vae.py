import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder import AutoEncoder
from models.variational_autoencoder_small_decoder import VariationalAutoEncoder
from transforms import train_transforms, test_transforms
import math
import os
from pathlib import Path

# Set device

device = "cpu"
pin_memory = True
use_amp = False
if torch.cuda.is_available():
    device = "cuda"
    use_amp=True

def train_vae(epochs, batch_size, bottleneck_size, output_file, dataset, learning_rate=1e-4):

    # Create the output path

    output_path = Path("output/") / output_file

    # Create the model

    model = VariationalAutoEncoder(bottleneck=bottleneck_size).to(device=device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Print number of parameters

    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}, trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training settings

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
    scheduler = ReduceLROnPlateau(optimizer, factor=1e-1, patience=5)

    reconstruction_loss = nn.MSELoss()
    def regularization_loss(mean, std):
        return torch.mean(-0.5*torch.sum((1 + torch.log(std.square()) - mean.square() - std.square()), dim=-1))

    def kl_annealing(epoch, epochs, beta):
        offset = int(epochs*20/100)
        return beta/(1+math.exp(-(epoch-offset)))
    
    min_val_loss = math.inf

    # Dataset & DataLoader Creation

    train_path = dataset / Path("train")
    val_path = dataset / Path("val")

    dataset_train = ImageFolder(train_path, transform=train_transforms)
    dataset_val = ImageFolder(val_path, transform=test_transforms)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    # Main training loop

    for epoch in range(epochs):
        running_reconstruction_loss = 0.0
        running_regularization_loss = 0.0

        model.train()

        for _, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # Pass our input through the model to get our output
                outputs, mean, std = model(inputs)

                loss = reconstruction_loss(outputs, inputs)

                current_reconstruction_loss = loss.item()
                running_reconstruction_loss += loss.item()

                # Variational encoders add a regularization term that computes the KL divergence between the encoder
                # distribution and the normal distribution
                loss += regularization_loss(mean, std)
                running_regularization_loss += loss.item() - current_reconstruction_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        model.eval()

        val_reconstruction_loss = 0.0
        val_regularization_loss = 0.0

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs, mean, std = model(inputs)

                loss = reconstruction_loss(outputs, inputs)
                val_current_reconstruction_loss = loss.item()
                val_reconstruction_loss += loss.item()

                # Variational encoders add a regularization term that computes the KL divergence between the encoder
                # distribution and the normal distribution
                loss += regularization_loss(mean, std)
                val_regularization_loss += loss.item() - val_current_reconstruction_loss
                
            # Save model weights and bottleneck size if improvement
            if (val_reconstruction_loss + val_regularization_loss) < min_val_loss:
                min_val_loss = val_reconstruction_loss + val_regularization_loss
                if torch.cuda.device_count() > 1:
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'bottleneck_size': bottleneck_size
                        }, output_path)
                else:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'bottleneck_size': bottleneck_size
                        }, output_path)

        
        scheduler.step(loss.item()/len(val_loader))
        
        print(f'Epoch: {epoch+1}/{epochs}, Train loss: {(running_reconstruction_loss/len(train_loader)):2f}/{(running_regularization_loss/len(train_loader)):2f}, Val loss: {(val_reconstruction_loss/len(val_loader)):2f}/{(val_regularization_loss/len(val_loader)):2f}')