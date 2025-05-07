# **Perceptron Algorithm and Visualization - Notes**

## **1. Libraries Import**

```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
```

* **`make_classification`**: Used to generate synthetic datasets for classification tasks.
* **`numpy`**: Used for numerical computations like dot products, arrays.
* **`matplotlib.pyplot`**: Used for data visualization (plotting).

---

## **2. Dataset Creation**

```python
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10
)
```

### **Parameters Explanation:**

* **`n_samples=100`**: Total 100 data points are generated.
* **`n_features=2`**: Two features for each data point.
* **`n_informative=1`**: Only one feature is useful for classification.
* **`n_redundant=0`**: No redundant (correlated) features.
* **`n_classes=2`**: Binary classification (labels 0 and 1).
* **`n_clusters_per_class=1`**: One cluster per class.
* **`random_state=41`**: Ensures reproducibility of the dataset.
* **`class_sep=10`**: Distance between classes, making them easily separable.

### **Output:**

* **`X.shape = (100, 2)`**: 100 data points with 2 features.
* **`y.shape = (100,)`**: 100 labels (0 or 1).

---

## **3. Perceptron Algorithm**

### **Perceptron Function**

```python
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0, 100)
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]

    return weights[0], weights[1:]
```

### **Steps:**

1. **Add Bias**:

   ```python
   X = np.insert(X, 0, 1, axis=1)
   ```

   Adds a column of 1s to `X` for the bias term.

2. **Initialize Weights**:

   ```python
   weights = np.ones(X.shape[1])
   ```

   Initializes weights for bias and features to `1`.

3. **Learning Rate**:

   ```python
   lr = 0.1
   ```

   Sets learning rate to 0.1 for weight updates.

4. **Training Loop (1000 iterations)**:

   ```python
   for i in range(1000):
       j = np.random.randint(0, 100)
       y_hat = step(np.dot(X[j], weights))
       weights = weights + lr * (y[j] - y_hat) * X[j]
   ```

   * Randomly selects data points and makes predictions (`y_hat`).
   * Updates weights using the formula:

     $$
     w = w + \eta \cdot (y - \hat{y}) \cdot X
     $$

     where $\eta$ is the learning rate, $y$ is the true label, and $\hat{y}$ is the predicted label.

5. **Return Weights**:

   ```python
   return weights[0], weights[1:]
   ```

   Returns the bias and feature weights.

---

## **4. Calculate Slope and Intercept for Decision Boundary**

```python
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])
```

* **Slope** (`m`): Calculated by dividing the feature weights.
* **Intercept** (`b`): Bias divided by the feature weight.

---

## **5. Generate Decision Boundary Line**

```python
x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b
```

* **`x_input`**: Generates 100 evenly spaced x-values from -3 to 3.
* **`y_input`**: Applies the equation of the line $y = mx + b$.

---

## **6. Plot Decision Boundary**

```python
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
```

### **Explanation**:

* **Figure size**: Sets the canvas size to 10x6 inches for better visualization.
* **Decision boundary**: Plots the line in red with thickness 3.
* **Data points**: Plots the points using the `winter` color map (blue-green) where `c=y` colors the points based on their class.
* **Y-axis limits**: Sets the y-axis range from -3 to 2 for a clear view.

---

### **Final Output:**

* The decision boundary line is drawn based on the perceptron’s learned weights, separating the two classes.
* Data points are plotted, and the line clearly divides the points based on their labels.

---

### **Summary Table**:

| **Step** | **Action**                           | **Output**                                 |
| -------- | ------------------------------------ | ------------------------------------------ |
| 1        | Import necessary libraries           | `sklearn`, `numpy`, `matplotlib`           |
| 2        | Create synthetic classification data | `X.shape = (100, 2)`, `y.shape = (100,)`   |
| 3        | Train perceptron                     | Weights for bias and features              |
| 4        | Calculate slope and intercept        | Line equation for decision boundary        |
| 5        | Generate line values for boundary    | `x_input`, `y_input` for decision boundary |
| 6        | Plot the decision boundary           | Red line and data points visualization     |

---

This is the **detailed notes** based on what you have learned today. I will save it in a **Markdown (.md)** file for you now.

Please give me a moment!
