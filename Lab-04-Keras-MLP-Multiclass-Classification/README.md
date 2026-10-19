# Keras MLP for Multiclass Classification

## Project Overview

This project implements a **Multi-Layer Perceptron (MLP)** using **TensorFlow/Keras** to perform **multiclass classification** on the **Iris dataset**.

The model is trained to classify iris flowers into three different classes.

---

## Technologies Used

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- Pandas

---

## Dataset

The project uses the **Iris Dataset** available in Scikit-learn.

Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Classes:
- Setosa
- Versicolor
- Virginica

---

## Model Architecture

- Input Layer (4 Features)
- Hidden Layer (16 Neurons, ReLU)
- Hidden Layer (8 Neurons, ReLU)
- Output Layer (3 Neurons, Softmax)

Optimizer:
- Adam

Loss Function:
- Sparse Categorical Crossentropy

---

## How to Run

### 1. Install the required packages

```bash
pip install tensorflow scikit-learn pandas
```

### 2. Run the program

```bash
python iris_classifier.py
```

---

## Output

The model trains for 50 epochs and displays:

- Training Accuracy
- Training Loss
- Test Accuracy

Example:

```
Test Accuracy: 1.0
```

---

## Project Structure

```
Deep-Learning-Lab-4/
│── iris_classifier.py
│── README.md
│── .gitignore
```

---

## Author

**Vaibhav**
