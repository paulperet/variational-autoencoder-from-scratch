import os

# Util: Create a toy dataset with the first 10 images of each class

train = os.listdir('./data/toy-dataset/train')

for classes in train:
    if os.path.isdir(f'./data/toy-dataset/train/{classes}'):
        images = os.listdir(f'./data/toy-dataset/train/{classes}')
        for image in images[10:]:
            os.remove(f'./data/toy-dataset/train/{classes}/{image}')