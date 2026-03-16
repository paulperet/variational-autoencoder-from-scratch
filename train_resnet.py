import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.resnet18 import ResNet18
from transforms import train_transforms, test_transforms
from torchmetrics import Accuracy

# Set device
device = "cpu"
pin_memory = True
if torch.cuda.is_available():
    device = "cuda"

num_classes=10

model = ResNet18(num_classes=num_classes).to(device=device)

#if torch.cuda.device_count() > 1:
#    model = nn.parallel.DistributedDataParallel(model)
print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}, trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

batch_size = 256

dataset_train = ImageFolder("./data/imagenette2-320/train", transform=train_transforms)
dataset_val = ImageFolder("./data/imagenette2-320/val", transform=test_transforms)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

use_amp = True

optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
scheduler = ReduceLROnPlateau(optimizer, factor=1e-1)
criterion = nn.CrossEntropyLoss()
epochs = int((60 * 10**4) / (len(train_loader) / batch_size))

train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

for epoch in range(epochs):
    running_loss = 0.0

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

        print(f'Epoch: {epoch}/{epochs}, Iteration: {i}/{len(train_loader)}, Loss: {running_loss/(i+1)}')
    

    scheduler.step(running_loss/len(train_loader))

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
    
    print(f'Epoch: {epoch}/{epochs}, Train loss: {running_loss/len(train_loader)}, Val loss: {val_total_loss}')
    print(f"Train accuracy: {train_accuracy.compute()}, Val accuracy: {val_accuracy.compute()}")

    train_accuracy.reset()
    val_accuracy.reset()