# Visual Encoders From Scratch

This repository explores the design, training, experiments and theorical foundations of auto-encoders and variational autoencoders (VAE). The goal is to show how the model complexity, dataset and training hyperparameters impact the model's performance. In addition, a clear and intuitive derivation of the ELBO is featured to explain the theorical grounding of variational encoders. 

The repository can serve as an introduction to autoencoders and VAEs and a guide on how to implement and train them in practice.

### The ResNet architecture

To illustrate our experiments, we will use a ResNet-18 CNN as our visual encoder, which as been slightly modified for our needs. The architecture for the encoder is as follow:

<img width="1379" height="1080" alt="resnet-18" src="https://github.com/user-attachments/assets/aadbef5e-c6b1-4c23-9b42-790015eb549d" />

The decoder is a reversed version of the encoder, where the convolutional layers are replaced by transposed convolutional layers.

#### Convolutional Layers

#### Transposed Convolution Layers

### Autoencoders

# References
[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
