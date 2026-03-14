import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.resnet18 import ResNet18
from transforms import train_transforms, test_transforms

model = ResNet18(num_classes=10)

batch_size = 64

dataset_train = ImageFolder("./data/toy-dataset/train", transform=train_transforms)
dataset_val = ImageFolder("./data/toy-dataset/val", transform=test_transforms)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

use_amp = True

optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
scheduler = ReduceLROnPlateau(optimizer, factor=1e-1)
criterion = nn.CrossEntropyLoss()
epochs = int((60 * 10**4) / (len(train_loader) / 256))

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)

        print(f'Epoch: {epoch}, Iteration: {i}, Loss: {running_loss/(i+1)}')
    

    scheduler.step(running_loss/len(train_loader))
    
    print(f'Epoch: {epoch}, Loss: {running_loss/len(train_loader)}')