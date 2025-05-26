# Exploding Gradient Problem in Deep Learning

## 🧠 What is the Exploding Gradient Problem?

Exploding gradients occur when the gradient values become very large during backpropagation. This usually happens in deep neural networks or recurrent neural networks (RNNs), causing the model to become unstable.

---

## 🔄 Where Does It Occur?

* Very deep neural networks
* RNNs with long sequences
* Optimizers using gradients (SGD, Adam)

---

## 🕮️ Mathematical View

Gradients are computed using the chain rule:

```math
\frac{dL}{dW} = \frac{dL}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdot \cdots \cdot \frac{dz_1}{dW}
```

If each term > 1, and n is large:

```math
\left| \frac{dL}{dW} \right| \approx (value > 1)^n
```

This exponential growth causes gradient explosion.

---

## 📉 Effect on Training

* Loss becomes NaN or infinity
* Weight values grow too large
* Model fails to learn or diverges
* Spiky or exploding loss curve

---

## 🔍 Symptoms

* Unstable or NaN loss
* Irregular model accuracy
* Extremely large weights

---

## ⚙️ Causes

1. Deep architecture
2. Poor weight initialization
3. High learning rate
4. No gradient clipping
5. Amplifying activation functions (like ReLU with large input)
6. Unnormalized input data

---

## 🚒 Solutions

### ✅ 1. Gradient Clipping

Limit gradient values:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### ✅ 2. Weight Regularization

Use L2 regularization (weight decay)

### ✅ 3. Proper Initialization

* Xavier (Glorot)
* He Initialization

### ✅ 4. Normalize Input Data

Mean = 0, Standard Deviation = 1

### ✅ 5. Use Smaller Learning Rate

### ✅ 6. Use LSTM/GRU

Gated architectures control gradient flow better.

### ✅ 7. Use Batch Normalization

Stabilizes input distribution.

---

## ⚡️ Exploding vs Vanishing Gradient

| Feature            | Exploding Gradient | Vanishing Gradient |
| ------------------ | ------------------ | ------------------ |
| Gradient Value     | Very large         | Very small         |
| Effect on Training | Diverges / NaN     | Stops learning     |
| Seen In            | Deep nets, RNNs    | Deep nets, RNNs    |
| Fix                | Clipping, BN, Init | ReLU, LSTM, Init   |

---

## 🌍 Real-World Scenarios

* RNNs for speech recognition
* GAN training
* Transformers (e.g., BERT, GPT) use clipping during training

---

## 📦 Tools in Libraries

* PyTorch: `clip_grad_norm_`, `clip_grad_value_`
* TensorFlow: `tf.clip_by_global_norm`

---

## ✅ Best Practices

* Normalize data
* Use GRU/LSTM for sequences
* Clip gradients
* Monitor loss curves
* Use proper initialization

---

## 🧠 Flashcards (Quick Q\&A)

**Q:** What is exploding gradient?
**A:** Gradient becomes too large, destabilizing training.

**Q:** Solution?
**A:** Gradient clipping.

**Q:** Common in which models?
**A:** Deep networks, RNNs.

**Q:** Sign of exploding gradient?
**A:** NaN loss, huge weights.

**Q:** Fix in PyTorch?
**A:** `torch.nn.utils.clip_grad_norm_()`
