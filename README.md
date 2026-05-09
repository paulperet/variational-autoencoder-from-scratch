# Visual Encoders From Scratch

This repository explores the design, training, experiments and theorical foundations of auto-encoders and variational autoencoders (VAE). The goal is to show how the model complexity, dataset and training hyperparameters impact the model's performance. In addition, a clear and intuitive derivation of the ELBO is featured to explain the theorical grounding of variational encoders. 

The repository can serve as an introduction to autoencoders and VAEs and a guide on how to implement and train them in practice.

<p align="center">
<img width="933" height="459" alt="model_comparison" src="https://github.com/user-attachments/assets/18b53f33-ad08-4ecd-88e1-487387921564" />
</p>

Above is an example of an autoencoder and variational autoencoder trained on the CelebA-HQ Dataset [1]. The dataset contains 30k high-quality images of celebrities. Both the autoencoder and variational encoders have been trained with a bottleneck size of 1024, for 100 epochs.

### Installation

Clone the repository & install dependencies
```bash
git clone https://github.com/paulperet/visual-encoders-from-scratch
cd visual-encoders-from-scratch
pip install -r requirements.txt
```

### The ResNet architecture

To illustrate our experiments, we will use a ResNet-18 CNN [2] as our visual encoder, which as been slightly modified for our needs. The architecture for the encoder is as follow:

<img width="1379" height="1080" alt="resnet-18" src="https://github.com/user-attachments/assets/aadbef5e-c6b1-4c23-9b42-790015eb549d" />

The decoder is a reversed version of the encoder, where the convolutional layers are replaced by transposed convolutional layers.

#### Convolutional Layers

#### Transposed Convolution Layers

### Autoencoders

### Variational Autoencoders

#### Directed Graphical Models

Example of a graphical model:

[EXAMPLE]

Observable nodes, from which we can access and measure data, are represented as shaded in directed graphical models. In contrast, latent nodes data cannot be accessed or measured.

Example of latent nodes:

[EXAMPLE]

#### ELBO

# References
[1] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)</br>
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
