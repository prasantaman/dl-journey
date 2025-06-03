## 📌 What is Data Augmentation?

> Data Augmentation is the process of **artificially increasing the size and diversity of a dataset** by applying random transformations to existing data.

---

## 🔍 Why Use Data Augmentation?

* Increases dataset size without collecting new data
* Reduces overfitting
* Helps model generalize better on unseen data
* Useful in deep learning where data is critical

---

## 🛠️ Common Data Augmentation Techniques for Images

| Technique           | Description                                 |
| ------------------- | ------------------------------------------- |
| Rotation            | Rotate image by a random angle (e.g., ±15°) |
| Horizontal Flip     | Flip image from left to right               |
| Vertical Flip       | Flip image from top to bottom               |
| Scaling             | Slightly resize image                       |
| Translation         | Shift image along X or Y axis               |
| Shearing            | Skew the image shape                        |
| Zoom                | Randomly zoom in or out                     |
| Brightness/Contrast | Modify brightness or contrast               |
| Noise Addition      | Add random noise to image                   |

---

## 🔄 Example in Python (Using Keras)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50)
```

---

## 🔢 Mathematical Insight

Let $x$ be the original image and $T$ be a transformation function:

$x' = T(x)$

Where $T$ is sampled from a set of transformations $\mathcal{T}$. This allows the model to learn features that are invariant to these transformations, effectively expanding the training distribution.

---

## ✅ Benefits Summary

* Increases dataset variety
* Helps in small datasets
* Prevents overfitting
* Boosts model robustness and accuracy

---

## ⚠️ Cautions

* Over-augmentation may distort class-specific features
* Make sure transformations are task-appropriate
* Some augmentations (like vertical flip) may not suit all use-cases

---

## 🔁 Flashcard Q\&A

**Q1. What is the main goal of data augmentation?**
A. To artificially increase training data size and diversity.

**Q2. Name three common augmentation techniques.**
A. Rotation, flipping, zooming.

**Q3. Can data augmentation prevent overfitting?**
A. Yes, by exposing model to diverse data.

**Q4. Is data augmentation applied during training or testing?**
A. During training only.
