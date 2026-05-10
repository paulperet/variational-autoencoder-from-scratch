from torchvision.transforms import transforms

# Augment train data
train_transforms = transforms.Compose([
    transforms.RandomCrop((224,224)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])