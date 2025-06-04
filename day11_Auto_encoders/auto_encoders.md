# Autoencoders: Complete Notes

---

## 1. **Introduction to Autoencoders**

Autoencoders are a type of **artificial neural network** used to learn efficient representations (encodings) of data, typically for **dimensionality reduction** or **feature learning**.

They are **unsupervised learning models**, which means they don't require labeled data.

**Goal**: Learn a compressed version (encoding) of the input data and then reconstruct the original input from that encoding.

---

## 2. **Architecture of Autoencoders**

Autoencoder has 3 main components:

### 1. **Encoder**

* Compresses input into a smaller representation.
* Maps input $x$ to latent representation $z$.
* $z = f(x)$

### 2. **Code (Latent Space)**

* Encoded, compressed version of input.
* Captures the most important features of the data.

### 3. **Decoder**

* Reconstructs original input from latent space.
* $\hat{x} = g(z)$

### Overall:

$\hat{x} = g(f(x))$

---

## 3. **Loss Function**

The loss measures the difference between input $x$ and output $\hat{x}$.

### Common loss functions:

* **Mean Squared Error (MSE)**:
  $L(x, \hat{x}) = \|x - \hat{x}\|^2$
* **Binary Crossentropy** (for binary data)

---

## 4. **Types of Autoencoders**

### 1. **Vanilla Autoencoder**

* Simple encoder-decoder structure with MLP layers.

### 2. **Sparse Autoencoder**

* Adds a sparsity constraint to the hidden units.
* Encourages model to activate only a few neurons.

### 3. **Denoising Autoencoder**

* Input is corrupted (e.g., noise added), model learns to reconstruct the original.
* Robust to noisy data.

### 4. **Variational Autoencoder (VAE)**

* Learns probability distribution of data.
* Latent space is continuous and allows for generative modeling.

### 5. **Convolutional Autoencoder**

* Uses Conv layers (instead of dense) for images.
* Better for spatial data like images.

### 6. **Contractive Autoencoder**

* Adds penalty to Jacobian of encoder to make it robust to small changes in input.

---

## 5. **Mathematics Behind Autoencoders**

### Encoder:

$z = \sigma(W_e x + b_e)$

### Decoder:

$\hat{x} = \sigma(W_d z + b_d)$

### Objective:

$\min \| x - \hat{x} \|^2$

In **VAE**:

* Encoder outputs mean ($\mu$) and standard deviation ($\sigma$) of Gaussian.
* Sample $z$ from $\mathcal{N}(\mu, \sigma^2)$
* KL Divergence is added to loss:
  $L = \text{Reconstruction Loss} + D_{KL}(q(z|x) || p(z))$

---

## 6. **Applications of Autoencoders**

* **Dimensionality reduction** (like PCA)
* **Noise removal** (denoising autoencoders)
* **Anomaly detection**
* **Image compression**
* **Image colorization**
* **Feature extraction**
* **Recommendation systems**
* **Generative models** (VAE, GANs)

---

## 7. **Advantages**

* Unsupervised learning
* Can be customized for images, sequences, etc.
* Learns features automatically
* Used for compression and noise reduction

---

## 8. **Limitations**

* Doesn’t generalize well if latent space is not smooth (except VAE)
* May just learn identity function if not properly regularized
* Requires tuning (architecture, layers, loss)

---

## 9. **Autoencoder vs PCA**

| Aspect           | PCA          | Autoencoder               |
| ---------------- | ------------ | ------------------------- |
| Type             | Linear       | Non-linear                |
| Basis            | Eigenvectors | Learned by network        |
| Learning         | Analytical   | Neural training           |
| Expressive Power | Limited      | High (deep architectures) |

---

## 10. **Code Example (PyTorch)**

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

---

## 11. **Future Scope**

* **Self-supervised learning**
* **3D Autoencoders** for volumetric data
* **Graph autoencoders**
* **Autoencoders in medicine**: MRI, genomics
* **Hybrid models**: Autoencoders + GANs (VAE-GAN)

---

## 12. **Flashcards (Q\&A)**

**Q: What is the main goal of an autoencoder?**
A: To learn a compressed representation of the input and then reconstruct the original input.

**Q: Which part compresses the data in an autoencoder?**
A: Encoder

**Q: What is the latent space in autoencoder?**
A: A compressed representation of the input.

**Q: What type of loss is used in autoencoders?**
A: Mean Squared Error or Binary Crossentropy.

**Q: What makes variational autoencoder different?**
A: It models data as a distribution and includes KL divergence in loss.

**Q: How does denoising autoencoder work?**
A: By learning to reconstruct original input from noisy input.

**Q: Can autoencoders be used for anomaly detection?**
A: Yes, by measuring reconstruction error.

**Q: What is the advantage of using convolutional autoencoders?**
A: Better performance on image data due to spatial feature extraction.

---
