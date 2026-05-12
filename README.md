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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create two folders for storing the dataset and saving the checkpoints
```bash
mkdir data
mkdir output
```

Dataset requirements:
- 224x224 or higher resolution
- 10k+ samples is better

Structure of the dataset:
```
.
├── data
│   ├── dataset_example
│   │   ├── train
│   │   │   ├── class1
│   │   │   │   ├── image.png
│   │   │   │   └── image.png
│   │   │   └── class2
│   │   │       ├── image.png
│   │   │       └── image.png
│   │   └── val
│   │       ├── class1
│   │       │    ├── image.png
│   │       │    └── image.png
│   │       └── class2
│   │            ├── image.png
│   │            └── image.png
```

Train a model:

```bash
python3 train.py [-h] [--model MODEL] --epochs EPOCHS --batch-size BATCH_SIZE --bottleneck-size BOTTLENECK_SIZE --output-file OUTPUT_FILE --dataset-folder DATASET_FOLDER [--learning-rate LEARNING_RATE] [--beta BETA]
```

Required arguments:
- model: ae or vae for autoencoder, variational autoencoder
- epochs: number of training epochs (full dataset pass)
- batch_size: number of samples per batch
- bottleneck size: dimensions of the latent space (lower -> more compression, blurry, higher -> better reconstruction, more details)
- output file: name of the model to save
- dataset folder: path to the dataset folder (for example data/dataset_example from above)

Optional arguments:
- learning rate: default 1e-3 for autoencoder and 1e-4 for VAE
- beta: weights the KL loss for the VAE, can be necessary for finding a good balance between reconstruction and organised/meaningful latent space (disentanglement), see [4]


### The ResNet architecture

To illustrate our experiments, we will use a ResNet-18 CNN [2] as our visual encoder, which as been slightly modified for our needs. The architecture for the encoder is as follow:

<img width="847" height="654" alt="resnet-18" src="https://github.com/user-attachments/assets/6dcb5022-4a86-47f3-8f5c-d94a40011ec5" />

The decoder is a reversed version of the encoder, where the convolutional layers are replaced by transposed convolutional layers.

### Autoencoders

### Variational Autoencoders

#### Directed Graphical Model

<p align="center">
<img width="640" height="239" alt="dgm" src="https://github.com/user-attachments/assets/67da06e0-e0a4-4b00-83a5-744afe8e729b" />
</p>

Observable nodes, from which we can access and measure data, are represented as shaded in directed graphical models. In contrast, latent nodes data are left unshaded and cannot be accessed or measured.

We represent the problem as a directed graphical model with an observed node $$x$$ (e.g. a dataset D of images of faces) and a latent node $$z$$ (latent or compressed representation of the data e.g. hair color, face orientation). According to this model, we can express the marginal as $$p(x,z) = p(z)p(x \mid z)$$.

#### Variational Inference

##### Intractible marginal

The goal is to perform inference, or to be able to recover the distribution of $$X$$ when observing a latent distribution $$Z$$. We express this as the posterior and it can be computed using Bayes' Rule:

$$p_{\theta} (z \mid x) = \frac{p_{\theta} (x \mid z)p_{\theta} (z)}{p_{\theta} (x)}$$

While we can compute the joint distribution, the marginal $$p_{\theta} (x)$$ is intractible. It cannot be easily computed analytically and the computational cost of approximating the denominator scales exponentially according to the domain of latent variables.

So instead of computing the posterior directly, we choose a surrogate function (simpler function) to approximate the original distribution:

$$q_{\phi} (z \mid x) \approx p_{\theta} (z \mid x)$$

##### Finding a surrogate via optimization

Thanks to this we can now express our problem as an optimization problem:

$$q^*(z \mid x) = argmin_{q_{\phi} (z \mid x) \in Q} (D_{KL}(q_{\phi} (z \mid x) \mathrel{\Vert} p_{\theta} (z \mid x))$$

The Kullback–Leibler divergence expresses the difference between two distributions. Our goal is to minimize this distance so that our surrogate function best capture the original distribution. But we still have a problem:

$$D_{KL}(q_{\phi} (z \mid x) \mathrel{\Vert} p_{\theta} (z \mid x)) = \mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z \mid x)}\right)\right] = \int_{z_{0}}...\int_{z_{D-1}}q_{\phi} (z \mid x)log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z \mid x)}\right)$$

We don't have the posterior $$p_{\theta} (z \mid x)$$! We only have the joint $$p_{\theta} (z, x)$$.

##### Rearranging the terms to isolate the marginal

$$
\begin{align*}
D_{KL}(q_{\phi} (z \mid x) \mathrel{\Vert} p_{\theta} (z \mid x)) 
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z \mid x)}\right)\right] \\
&= \int_{z_{0}}...\int_{z_{D-1}}q_{\phi} (z \mid x)log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z \mid x)}\right) \\
&= \int_{z_{0}}...\int_{z_{D-1}}q_{\phi} (z \mid x)log\left(\frac{q_{\phi} (z \mid x)p_{\theta} (x)}{p_{\theta} (z,x)}\right) \\
&= \int_{z_{0}}...\int_{z_{D-1}}q_{\phi} (z \mid x)log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z,x)}\right) \int_{z_{0}}...\int_{z_{D-1}}q_{\phi} (z \mid x)log\left(p_{\theta} (x))\right) \\
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z, x)}\right)\right] + \mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log(p_{\theta} (x))\right] \\
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{q_{\phi} (z \mid x)}{p_{\theta} (z, x)}\right)\right] + log(p_{\theta} (x)) \\
&= -\mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{p_{\theta} (z, x)}{q_{\phi} (z \mid x)}\right)\right] + log(p_{\theta} (x)) \\
\end{align*}
$$

Lets denote $$\mathbb{E_{z \sim q_{\phi} (z \mid x)}}\left[log\left(\frac{p_{\theta} (z, x)}{q_{\phi} (z \mid x)}\right)\right]$$ by $$\mathcal{L}(q)$$

We have $$D_{KL} = -\mathcal{L}(q) + log(p_{\theta} (x))$$

We know that:
- KL divergence is greater or equal to 0.
- $$p_{\theta} (x)$$ is between 0 and 1 and the log of that is equal or less than 0.

Hence, to fullfill these conditions, we can infer that $$\mathcal{L}(q)$$ has to be negative, and is named the evidence lower bound. Therefore, because $$log(p_{\theta} (x))$$ is constant (our dataset), minimizing the KL divergence is equivalent to maximizing the ELBO:

$$
\begin{align*}
q^* (z) &= argmin_{q_{\phi} (z \mid x) \in Q} (KL(q_{\phi} (z \mid x) \mathrel{\Vert} p_{\theta} (z \mid x))) \\
&= argmax_{q_{\phi} (z \mid x) \in Q} (\mathcal{L}(q))
\end{align*}
$$

Expanding $$\mathcal{L}(q)$$, we have:

$$
\begin{align*}
\mathcal{L}(q) &= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[-log(q_{\phi} (z \mid x)) + log(p_{\theta} (z, x))] \\
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[-log(q_{\phi} (z \mid x)) + log(p_{\theta} (x \mid z)(p_{\theta} (z))] \\
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[-log(q_{\phi} (z \mid x)) + log(p_{\theta} (x \mid z)) + log(p_{\theta} (z)] \\
&= \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[-log(q_{\phi} (z \mid x)) + log(p_{\theta} (z))] + \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[log(p_{\theta} (x \mid z)] \\
&= -D_{KL}(q_{\phi} (z \mid x)) \mathrel{\Vert} p_{\theta} (z)) + \mathbb{E_{z \sim q_{\phi} (z \mid x)}}[log(p_{\theta} (x \mid z)] \\
\end{align*}
$$

##### Estimating the gradient

In practice, we will choose $$q_{\phi} (z \mid x)$$ and $$p_{\theta} (Z)$$ to be normal or bernoulli distributions, depending on our data. This means that we can easily integrate the KL term analytically. Therefore, we only need to estimate the reconstruction term $$\mathbb{E_{z \sim q_{\phi} (z \mid x)}}[log(p_{\theta} (x \mid z)]$$. We can use Monte-Carlo sampling to find a good estimate:

$$\mathcal{L}(q) = -D_{KL} (q_{\phi} (z \mid x)) \mathrel{\Vert} p_{\theta} (z)) + \frac{1}{L} \sum_{l=0}^L log(p_{\theta} (x \mid z)$$

We calculate the gradient over a minibatch:

$$\mathcal{L}(\theta, \phi; x) \approx \mathcal{L^M}(\theta, \phi; X^M) = \frac{N}{M} \sum_{i=1}^M \mathcal{L}(\theta, \phi; x^{(i)})$$

Often, the estimation over a large enough batch (M>100) while only sampling a single $$p_{\theta} (x \mid z)$$ yields a sufficiently good estimate, so we can use this expression in practice:

$$\mathcal{L}(q) = -D_{KL}(q_{\phi} (z \mid x)) \mathrel{\Vert} p_{\theta} (z)) + log(p_{\theta} (x \mid z)$$

##### Simplifying the loss to a practical loss function

It is common to use gaussian distributions for modelling colored images, and bernoulli for black & white images. We can therefore simplify the KL term to:

$$-D_{KL} (q_{\phi}(z) \mathrel{\Vert} p_{\theta}(z)) = \frac{1}{2} \sum_{j=1}^J (1 + log((\sigma^2)) - (\mu)^2 - (\sigma)^2)$$

And the reconstruction term to:

$$log(p_{\theta} (x \mid z)) \approx \sum_{j=1}^J (x_{i}^{(j)} - \mu_{i}^{(j)})$$

# References
Research papers:

[1] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) </br>
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) </br>
[3] [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) </br>
[4] [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) </br>

Videos:

[What is a latent variable?](https://www.youtube.com/watch?v=SNeC_SrbNZw) </br>
[Introduction to Directed Graphical Models | Implementation in TensorFlow Probability
](https://www.youtube.com/watch?v=yBc01ZeaFxw) </br>
[Variational Inference | Evidence Lower Bound (ELBO) | Intuition & Visualization](https://www.youtube.com/watch?v=HxQ94L8n0vU&list=PLt5BUXqHz_OlQwyOSAT2cOvCcdBWVKBK5) </br>
[The challenges in Variational Inference (+ visualization)](https://www.youtube.com/watch?v=gV1NWMiiAEI&list=PLt5BUXqHz_OlQwyOSAT2cOvCcdBWVKBK5&index=2)


