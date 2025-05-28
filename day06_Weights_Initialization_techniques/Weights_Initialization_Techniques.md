# Weight Initialization Techniques in Deep Learning

Weight initialization plays a crucial role in the training of deep neural networks. Poor initialization can cause the gradients to vanish or explode, slowing down or completely halting training.

---

## 1. Zero Initialization

```python
W = np.zeros((shape))
```

**Description**: All weights are initialized to zero.

**Problem**:

* Fails to break symmetry. All neurons compute the same output.
* Gradients remain the same during backpropagation.

**✅ Use**: Only for initializing biases.

---

## 2. Random Initialization

```python
W = np.random.randn(shape) * 0.01
```

**Description**: Weights initialized with small random values.

**Pros**:

* Breaks symmetry.

**Cons**:

* Can still cause vanishing/exploding gradients if not scaled properly.

---

## 3. Xavier (Glorot) Initialization

```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
```

**Best for**: Sigmoid or Tanh activation functions.

**Variants**:

* **Normal**: $W \sim \mathcal{N}(0, 1/n_{in})$
* **Uniform**: $W \sim \mathcal{U}(-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})})$

---

## 4. He (Kaiming) Initialization

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

**Best for**: ReLU and Leaky ReLU

**Formula**:

* $W \sim \mathcal{N}(0, 2/n_{in})$

---

## 5. LeCun Initialization

```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
```

**Best for**: SELU activation functions

**Formula**:

* $W \sim \mathcal{N}(0, 1/n_{in})$

---

## 6. Orthogonal Initialization

**Description**: Uses orthogonal matrices (preserves variance).

**Best for**: Deep networks and RNNs

**Pros**:

* Maintains variance of activations
* Stabilizes training

---

## 7. Sparse Initialization

**Description**: Most weights are initialized to zero; few non-zero values.

**Best for**: Large and sparse models

---

## Activation vs Initialization Table

| Activation     | Recommended Initialization |
| -------------- | -------------------------- |
| Sigmoid/Tanh   | Xavier                     |
| ReLU/LeakyReLU | He                         |
| SELU           | LeCun                      |

---

## Summary Table

| Technique       | Formula                            | Best for      |
| --------------- | ---------------------------------- | ------------- |
| Zero            | All zeros                          | ❌ (Don't use) |
| Random Small    | Random \* 0.01                     | Basic usage   |
| Xavier (Glorot) | Normal: $\mathcal{N}(0, 1/n_{in})$ | Sigmoid/Tanh  |
| He (Kaiming)    | Normal: $\mathcal{N}(0, 2/n_{in})$ | ReLU          |
| LeCun           | Normal: $\mathcal{N}(0, 1/n_{in})$ | SELU          |
| Orthogonal      | Orthogonal Matrix scaled by gain   | RNNs          |
| Sparse          | Mostly 0, few random values        | Sparse nets   |

---

## Flashcards

* **Q:** Why is zero initialization bad for weights?
  **A:** It causes symmetry — all neurons learn the same thing.

* **Q:** Which initialization is best for ReLU?
  **A:** He Initialization.

* **Q:** Xavier initialization is best suited for which activations?
  **A:** Sigmoid or Tanh.

* **Q:** What is the formula for He initialization?
  **A:** $\mathcal{N}(0, 2/n_{in})$

* **Q:** Which initialization keeps variance constant for SELU?
  **A:** LeCun initialization.
