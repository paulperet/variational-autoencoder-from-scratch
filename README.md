# Visual Encoders From Scratch

This repository explores the design, training, experiments and theorical foundations of auto-encoders and variational autoencoders (VAE). The goal is to show how the model complexity, dataset and training hyperparameters impact the model's performance. In addition, a clear and intuitive derivation of the ELBO is featured to explain the theorical grounding of variational encoders. 

The repository can serve as an introduction to autoencoders and VAEs and a guide on how to implement and train them in practice.

<p align="center">
<img width="1615" height="749" alt="model_comparison-3" src="https://github.com/user-attachments/assets/ffcf2247-5007-4df1-8953-3508509464af" />
</p>

Above is an example of an autoencoder and variational autoencoder trained on the CelebA-HQ Dataset [1]. The dataset contains 30k high-quality images of celebrities. Both the autoencoder and variational encoders have been trained with a bottleneck size of 1024, for 100 epochs. As we can observe, the autoencoder is able to reconstruct more precisely, but is unable to generate any coherent image du to the disorganized latent space. The VAE has less precise reconstructions (more blur) but is able to generate new samples.

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

#### Variational Inference

##### Intractible marginal

We represent the problem as a directed graphical model with an observed node $$X$$ (e.g. a dataset D of images of faces) and a latent node $$Z$$ (latent or compressed representation of the data e.g. hair color, face orientation). The goal is to perform inference, or to be able to recover the distribution of $$X$$ when observing a latent distribution $$Z$$. We express this as the posterior and it can be computed using Bayes' Rule:

$$P(X|X=D) = \frac{P(X=D|Z)P(Z)}{P(X=D)}$$

While we can compute the joint distribution, the marginal $$P(X=D)$$ is intractible. It cannot be easily computed analytically and the computational cost of approximating the denominator scales exponentially according to the domain of latent variables.

So instead of computing the posterior directly, we choose a surrogate function (simpler function) to approximate the original distribution:

$$q(Z) \approx p(Z|X=D)$$

##### Finding a surrogate via optimization

Thanks to this we can now express our problem as an optimization problem:

$$q^*(Z) = argmin_{q(z) \in Q} (KL(q(Z) || p(Z|X=D)$$

The Kullback–Leibler divergence expresses the difference between two distributions. Our goal is to minimize this distance so that our surrogate function best capture the original distribution. But we still have a problem:

$$KL(q(Z) || p(Z | X=D)) = \mathbb{E_{z \sim q(Z)}}[log(\frac{q(Z)}{p(Z|X=D)})] = \int_{z_{0}}...\int_{z_{D-1}}q(Z)log\frac{q(Z)}{p(Z|X=D)}$$

We don't have the posterior $$P(Z|X=D)$$! We only have the joint $$P(Z, X=D)$$.

##### Rearranging the terms to isolate the marginal

$$
\begin{align*}
KL(q(Z) || p(Z | X=D)) = \mathbb{E_{z \sim q(Z)}}[log(\frac{q(Z)}{p(Z|X=D)})] &= \int_{z_{0}}...\int_{z_{D-1}}q(Z)log\frac{q(Z)}{p(Z|X=D)} \\
&= \int_{z_{0}}...\int_{z_{D-1}}q(Z)log\frac{q(Z)p(X=D)}{p(Z,X=D)} \\
&= \int_{z_{0}}...\int_{z_{D-1}}q(Z)log\frac{q(Z)}{p(Z,X=D)} \int_{z_{0}}...\int_{z_{D-1}}q(Z)log(p(X=D)) \\
&= \mathbb{E_{z \sim q(Z)}}[log(\frac{q(Z)}{p(Z, X=D)})] + \mathbb{E_{z \sim q(Z)}}[log(p(X=D))] \\
&= \mathbb{E_{z \sim q(Z)}}[log(\frac{q(Z)}{p(Z, X=D)})] + log(p(X=D)) \\
\end{align*}
$$

# References
[1] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)</br>
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
