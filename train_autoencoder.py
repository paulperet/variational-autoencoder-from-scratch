import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder import AutoEncoder
from transforms import train_transforms, test_transforms

# Set device

device = "cpu"
pin_memory = True
if torch.cuda.is_available():
    device = "cuda"

# Create the model

model = AutoEncoder(bottleneck=1024).to(device=device)

# Print number of parameters

print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}, trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training settings

epochs = 10
batch_size = 256
use_amp = True

optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
scheduler = ReduceLROnPlateau(optimizer, factor=1e-1, patience=5)
criterion = nn.MSELoss()

# Dataset & DataLoader Creation

dataset_train = ImageFolder("./data/imagenette2-320/train", transform=train_transforms)
dataset_val = ImageFolder("./data/imagenette2-320/val", transform=test_transforms)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

# Main training loop

for epoch in range(epochs):
    running_loss = 0.0

    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)

        #print(f'Epoch: {epoch+1}/{epochs}, Iteration: {i+1}/{len(train_loader)}, Loss: {running_loss/(i+1)}')

    model.eval()

    val_total_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_total_loss += loss.item()
    
    scheduler.step(val_total_loss/len(val_loader))
    
    print(f'Epoch: {epoch+1}/{epochs}, Train loss: {running_loss/len(train_loader)}, Val loss: {val_total_loss/len(val_loader)}')