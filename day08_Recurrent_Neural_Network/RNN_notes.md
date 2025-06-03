## Recurrent Neural Network (RNN) – Full Detailed Notes

---

### 🔹 Introduction

RNN (Recurrent Neural Network) is a type of neural network designed for **sequential data**, like time series, text, audio, etc.

Unlike traditional networks (ANN/CNN) that assume inputs are independent, RNN maintains a **memory** of previous inputs using hidden states.

---

### 🔹 Real-life Examples

* Text generation
* Language translation
* Speech recognition
* Stock price prediction
* Music generation
* Sentiment analysis

---

### 🔹 Architecture

```
      xt           xt+1         xt+2
       ↓             ↓             ↓
   ┌───────┐     ┌───────┐     ┌───────┐
   │       │     │       │     │       │
   │   ht  │◄────│  ht+1 │◄────│  ht+2 │◄── ...
   │       │     │       │     │       │
   └───────┘     └───────┘     └───────┘
       ↓             ↓             ↓
      yt           yt+1         yt+2
```

* `xt` = input at time step `t`
* `ht` = hidden state at time `t` (memory)
* `yt` = output at time `t`

---

### 🔹 Working (Forward Pass)

At time step `t`:

```math
ht = tanh(Wxh * xt + Whh * ht-1 + bh)
yt = Why * ht + by
```

* `Wxh`: Input weight matrix
* `Whh`: Hidden-to-hidden weight matrix
* `Why`: Hidden-to-output matrix
* `bh`, `by`: Biases
* `tanh`: Activation function

---

### 🔹 Backpropagation Through Time (BPTT)

* Backpropagation is done **through time** (over multiple timesteps)
* Gradients calculated from end to start

#### 🔻 Problem: Vanishing/Exploding Gradient

* Happens in long sequences
* Gradients shrink (vanish) or grow too large (explode)
* Solution: Use LSTM/GRU

---

### 🔹 Types of RNN Architectures

| Type         | Input    | Output   | Example              |
| ------------ | -------- | -------- | -------------------- |
| One to One   | Single   | Single   | Image classification |
| One to Many  | Single   | Sequence | Music generation     |
| Many to One  | Sequence | Single   | Sentiment analysis   |
| Many to Many | Sequence | Sequence | Language translation |

---

### 🔹 Variants of RNN

#### ✅ LSTM (Long Short-Term Memory)

* Designed for long-term dependencies
* Uses gates:

  * Forget Gate
  * Input Gate
  * Output Gate

#### ✅ GRU (Gated Recurrent Unit)

* Simpler than LSTM
* Only two gates: Update Gate and Reset Gate

---

### 🔹 Applications in Data Science

| Domain     | Use Case                   |
| ---------- | -------------------------- |
| NLP        | Text generation, sentiment |
| Finance    | Stock prediction           |
| Healthcare | Patient monitoring         |
| Speech     | Voice assistants           |
| IoT        | Sensor data analysis       |

---

### 🔹 Advantages

* Remembers past inputs
* Suitable for sequence problems
* Enhanced by LSTM/GRU

---

### 🔹 Limitations

* Vanishing gradient
* Slow training
* No parallelization across time

---

### 🔹 Python Code Example (Using Keras)

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(50, input_shape=(timesteps, features)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

---

### 🔹 RNN vs CNN vs ANN vs LSTM

| Feature   | ANN    | CNN    | RNN       | LSTM       |
| --------- | ------ | ------ | --------- | ---------- |
| Data Type | Static | Grid   | Sequence  | Sequence   |
| Memory    | No     | No     | Yes       | Yes (long) |
| Speed     | Fast   | Medium | Slow      | Slower     |
| Use Case  | Tables | Images | Text/Time | Long Text  |

---

### 📌 Summary

* RNN learns sequence patterns
* Maintains hidden state (memory)
* Uses BPTT to learn
* Replaced by LSTM/GRU for long sequences

---

### 📖 Flashcards

```
Q: What is the key feature of an RNN?
A: It uses memory (hidden state) to capture dependencies in sequential data.

Q: Why does RNN suffer from vanishing gradient?
A: Because repeated multiplication of small derivatives in long sequences causes gradients to shrink to zero.

Q: What replaces RNN in modern systems for long sequences?
A: LSTM and GRU.

Q: What is the role of the tanh function in RNN?
A: It adds non-linearity and keeps outputs between -1 and 1.

Q: What is BPTT?
A: Backpropagation Through Time – used to train RNNs over time steps.

Q: Main use of RNN in NLP?
A: Text generation, translation, sentiment analysis.
```
