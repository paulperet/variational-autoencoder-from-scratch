import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder import AutoEncoder
from models.variational_autoencoder import VariationalAutoEncoder
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

def train_autoencoder(epochs, batch_size, bottleneck_size, output_file, dataset):

    # Create the output path

    output_path = Path("output/") / output_file

    # Create the model

    model = AutoEncoder(bottleneck=bottleneck_size).to(device=device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Print number of parameters

    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}, trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training settings

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
    scheduler = ReduceLROnPlateau(optimizer, factor=1e-1, patience=5)

    reconstruction_loss = nn.MSELoss(reduction='none')
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
        running_loss = 0.0

        model.train()

        for _, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # Pass our input through the model to get our output
                outputs = model(inputs)

                loss = reconstruction_loss(outputs, inputs).sum([1,2,3]).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            optimizer.zero_grad(set_to_none=True)

            #print(f'Epoch: {epoch+1}/{epochs}, Iteration: {i+1}/{len(train_loader)}, Loss: {running_loss/(i+1)}')

        model.eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = reconstruction_loss(outputs, inputs).sum([1,2,3]).mean()

                val_total_loss += loss.item()
                
                # Save model weights and bottleneck size if improvement
                if val_total_loss < min_val_loss:
                    min_val_loss = val_total_loss
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

        
        scheduler.step(val_total_loss/len(val_loader))
        
        print(f'Epoch: {epoch+1}/{epochs}, Train loss: {running_loss/len(train_loader)}, Val loss: {val_total_loss/len(val_loader)}')