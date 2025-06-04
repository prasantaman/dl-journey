## 📘 Activation Functions in Neural Networks – Full Detail

Activation functions are mathematical operations applied to the output of a neural network node (neuron) to introduce **non-linearity** into the model.

---

### 🔹 Why Activation Functions?

* Without activation functions, a neural network is just a **linear regression model**, regardless of the number of layers.
* Activation functions allow the model to learn **complex patterns**.

---

### 🔹 Types of Activation Functions

#### 1. **Linear Activation Function**

* Formula: $f(x) = x$
* Output is not bounded; model remains linear.
* **Problem**: Can’t capture complex non-linear patterns.
* **Use Case**: Rarely used in hidden layers.

---

#### 2. **Sigmoid Function**

* Formula: $f(x) = \frac{1}{1 + e^{-x}}$
* Output Range: (0, 1)
* Used in binary classification.
* **Advantages**:

  * Smooth gradient
  * Output can be interpreted as probability
* **Disadvantages**:

  * Vanishing gradient problem
  * Output not zero-centered

---

#### 3. **Tanh (Hyperbolic Tangent)**

* Formula: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
* Output Range: (-1, 1)
* **Advantages**:

  * Zero-centered output
  * Stronger gradients than sigmoid
* **Disadvantages**:

  * Still suffers from vanishing gradient for large |x|

---

#### 4. **ReLU (Rectified Linear Unit)**

* Formula: $f(x) = \max(0, x)$
* Output Range: \[0, ∞)
* **Advantages**:

  * Computationally efficient
  * Reduces likelihood of vanishing gradients
* **Disadvantages**:

  * Dying ReLU problem (neurons become inactive)

---

#### 5. **Leaky ReLU**

* Formula: $f(x) = \max(\alpha x, x)$, where $\alpha \approx 0.01$
* Fixes dying ReLU by allowing a small gradient for x < 0

---

#### 6. **Parametric ReLU (PReLU)**

* Similar to Leaky ReLU but $\alpha$ is learned during training

---

#### 7. **ELU (Exponential Linear Unit)**

* Formula:

  $$f(x) = \begin{cases}
  x & \text{if } x > 0 \\
  \alpha(e^x - 1) & \text{if } x \leq 0
  \end{cases} \]
  $$
* Smooths transition from negative to positive

---

#### 8. **Swish (by Google)**

* Formula: $f(x) = x \cdot \sigma(x)$
* Combines linearity and non-linearity
* Performs better in deep models

---

#### 9. **Softmax Function**

* Converts raw scores into probabilities that sum to 1
* Formula for class i:

  $$
  \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
  $$
* **Use Case**: Multi-class classification (output layer)

---

### 🔹 Choosing the Right Activation Function

| Layer Type         | Recommended Activation |
| ------------------ | ---------------------- |
| Hidden Layers      | ReLU / Leaky ReLU      |
| Binary Output      | Sigmoid                |
| Multi-class Output | Softmax                |
| RNNs               | Tanh / ReLU            |

---

### 🔹 Issues with Activation Functions

* **Vanishing Gradient**: Sigmoid/Tanh
* **Dying ReLU**: ReLU (fixed by Leaky ReLU/PReLU)
* **Computational cost**: ELU, Swish are more expensive

---

### 🔹 Summary Table

| Function   | Range   | Non-linearity | Issues                    | Use Case                   |
| ---------- | ------- | ------------- | ------------------------- | -------------------------- |
| Sigmoid    | (0,1)   | Yes           | Vanishing gradient        | Binary classification      |
| Tanh       | (-1,1)  | Yes           | Vanishing gradient        | RNNs                       |
| ReLU       | \[0, ∞) | Yes           | Dying neurons             | Hidden layers              |
| Leaky ReLU | (-∞, ∞) | Yes           | Still dying sometimes     | Variant of ReLU            |
| PReLU      | (-∞, ∞) | Yes           | Extra parameters          | Learnable ReLU variant     |
| ELU        | (-α, ∞) | Yes           | Computationally expensive | Deep learning models       |
| Swish      | (-∞, ∞) | Yes           | Newer, heavier to compute | Advanced deep models       |
| Softmax    | (0,1)   | Yes           | For classification only   | Multi-class classification |

---

## 📌 Q\&A Flashcards

**Q1: Why is ReLU preferred in hidden layers?**
A: Because it is simple, fast, and avoids vanishing gradients.

**Q2: Which function outputs values between 0 and 1?**
A: Sigmoid

**Q3: What’s the issue with sigmoid and tanh?**
A: They suffer from vanishing gradient problem.

**Q4: Which function is used in multi-class classification?**
A: Softmax

**Q5: What fixes dying ReLU problem?**
A: Leaky ReLU or PReLU
