import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.resnet18 import ResNet18
from utils.transforms import train_transforms, test_transforms
from torchmetrics import Accuracy
import os
from pathlib import Path
import math

# Set device
device = "cpu"
pin_memory = True
use_amp = False
if torch.cuda.is_available():
    device = "cuda"
    use_amp=True

def train_resnet(epochs, batch_size, output_file, dataset, learning_rate):
    # Create the output path
    output_path = Path("output/") / output_file

    train_path = dataset / Path("train")
    val_path = dataset / Path("val")

    num_classes = len(os.listdir(train_path))
    print(f"Number of classes: {num_classes}")

    model = ResNet18(num_classes=num_classes).to(device=device)

    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model)
    
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}, trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset_train = ImageFolder(train_path, transform=train_transforms)
    dataset_val = ImageFolder(val_path, transform=test_transforms)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
    scheduler = ReduceLROnPlateau(optimizer, factor=1e-1, patience=5)
    criterion = nn.CrossEntropyLoss()

    train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    min_val_loss = math.inf

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_accuracy.update(outputs, labels)

            optimizer.zero_grad(set_to_none=True)

        model.eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                val_accuracy.update(outputs, labels)

                # Save model weights and bottleneck size if improvement
                if val_total_loss < min_val_loss:
                    min_val_loss = val_total_loss
                    if torch.cuda.device_count() > 1:
                        torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, output_path)
                    else:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, output_path)
        
        scheduler.step(val_total_loss/len(val_loader))
        
        print(f'Epoch: {epoch+1}/{epochs}, Train loss: {running_loss/len(train_loader)}, Val loss: {val_total_loss/len(val_loader)}')
        print(f"Train accuracy: {train_accuracy.compute()}, Val accuracy: {val_accuracy.compute()}")

        train_accuracy.reset()
        val_accuracy.reset()