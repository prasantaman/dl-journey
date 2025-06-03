
## 🔹 Gated Recurrent Unit (GRU) – Full Notes

---

### 🔹 What is GRU?

**GRU (Gated Recurrent Unit)** is a type of **Recurrent Neural Network (RNN)** architecture designed to handle **sequential data** and **long-term dependencies**, just like LSTM, but with a **simpler structure** and **faster training**.

It was introduced by **Cho et al., 2014** as a **lightweight alternative** to LSTM.

---

### 🔹 Why GRU?

RNNs suffer from **vanishing gradients** in long sequences.

LSTMs fix this using gates, but they are **computationally expensive**.

**GRUs use fewer gates and parameters**, offering:

* Faster training
* Less memory usage
* Comparable performance to LSTM

---

### 🔹 GRU Architecture

GRU has **two gates**:

1. **Update Gate (zₜ)** – Controls how much of the past information to carry forward.
2. **Reset Gate (rₜ)** – Decides how much of the past to forget.

---

### 🔹 GRU Formulas

Let:

* $x_t$ = input at time t
* $h_{t-1}$ = previous hidden state
* $h_t$ = current hidden state

#### **1. Update Gate (zₜ)**

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$

→ Controls how much past info to keep.

---

#### **2. Reset Gate (rₜ)**

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

→ Controls how much past info to forget.

---

#### **3. Candidate Activation (ĥₜ)**

$$
\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])
$$

→ Generates new memory using reset gate.

---

#### **4. Final Output (hₜ)**

$$
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
$$

→ Mix of old and new memory.

---

### 🔹 GRU vs LSTM vs RNN

| Feature     | RNN                      | LSTM                      | GRU                    |
| ----------- | ------------------------ | ------------------------- | ---------------------- |
| Gates       | None                     | 3 (input, forget, output) | 2 (update, reset)      |
| Memory Cell | No                       | Yes                       | No                     |
| Performance | Low (vanishing gradient) | High (slow to train)      | High (faster training) |
| Parameters  | Fewer                    | Many                      | Fewer than LSTM        |

---

### 🔹 Diagram of GRU (Description for drawing)

```
           ┌─────────────┐
 xₜ ─────▶│  Update Gate │───▶ zₜ
          └─────────────┘
                │
 xₜ ─────▶┐     ▼
          ▼  ┌─────────────┐
        hₜ₋₁ │  Reset Gate │───▶ rₜ
             └─────────────┘
                  │
                  ▼
          ┌──────────────────┐
          │ Candidate Memory │ (h̃ₜ)
          └──────────────────┘
                  │
                  ▼
        ┌────────────────────────┐
        │ Final Output (hₜ)      │
        └────────────────────────┘
```

You can draw this with arrows showing flow from $x_t$, $h_{t-1}$ into the gates and then into final output $h_t$.

---

### 🔹 Advantages of GRU

* **Faster Training**: Fewer gates = fewer parameters = faster.
* **Efficient for Small Datasets**
* **Good for Medium-Length Sequences**
* **Simpler than LSTM, better than RNN**

---

### 🔹 Disadvantages

* May **underperform on very long sequences** compared to LSTM.
* Cannot **control memory output** as finely as LSTM's output gate.

---

### 🔹 Applications of GRU

* **Chatbots**
* **Machine Translation**
* **Speech Recognition**
* **Stock Price Prediction**
* **Sentiment Analysis**

---

### 📌 Summary Table

| Term        | Meaning                                                  |
| ----------- | -------------------------------------------------------- |
| GRU         | Simplified LSTM with only update and reset gates         |
| Update Gate | Decides what information to keep                         |
| Reset Gate  | Decides what information to forget                       |
| h̃ₜ         | New candidate memory generated using reset-modified past |
| hₜ          | Final memory using update gate to blend old and new info |

---

### 📖 Flashcards

```
Q: What does GRU stand for?
A: Gated Recurrent Unit

Q: How many gates are there in GRU?
A: Two – Update Gate and Reset Gate

Q: Which gate in GRU controls forgetting?
A: Reset Gate (rₜ)

Q: Which gate in GRU controls memory blending?
A: Update Gate (zₜ)

Q: How is GRU different from LSTM?
A: GRU has 2 gates (vs 3 in LSTM), no memory cell, and is simpler/faster.
```
