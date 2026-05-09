# Visual Encoders From Scratch

This repository explores the design, training, experiments and theorical foundations of auto-encoders and variational autoencoders (VAE). The goal is to show how the model complexity, dataset and training hyperparameters impact the model's performance. In addition, a clear and intuitive derivation of the ELBO is featured to explain the theorical grounding of variational encoders. 

The repository can serve as an introduction to autoencoders and VAEs and a guide on how to implement and train them in practice.

### The ResNet architecture

To illustrate our experiments, we will use a ResNet-18 CNN as our visual encoder, which as been slightly modified for our needs. The architecture is as follow:

<img width="1379" height="1080" alt="resnet-18" src="https://github.com/user-attachments/assets/bbb4c0fa-7e89-4756-9cd5-a03f71ff29f4" />

# References
[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
