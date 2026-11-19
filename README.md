# 🧠 Artificial Neural Networks & Deep Learning — Lab Coursework

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Comprehensive academic and practical laboratory repository covering fundamental perceptrons, feedforward neural networks, multi-layer perceptrons (MLPs), and convolutional neural networks (CNNs).**

---

## 📑 Coursework Overview & Syllabus Mapping

This monorepo consolidates all 7 practical laboratory assignments into an organized structure with standardized implementations in **NumPy**, **PyTorch**, and **TensorFlow / Keras**.

| Lab | Directory | Core Topic | Tech Stack | Key Concepts & Models |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [`Lab-01-Perceptron-Irrigation-Decision`](./Lab-01-Perceptron-Irrigation-Decision) | Irrigation Decision Automation | NumPy, Python | Single-layer Perceptron, Step Function, Binary Decision Boundary |
| **02** | [`Lab-02-Single-Neuron-Perceptron-Loan-Approval`](./Lab-02-Single-Neuron-Perceptron-Loan-Approval) | Credit & Loan Approval Classifier | NumPy, Python | Single-Neuron Perceptron, Weight Updates, AND Gate Analogy |
| **03** | [`Lab-03-Feedforward-NN-PyTorch`](./Lab-03-Feedforward-NN-PyTorch) | Feedforward Neural Network (FNN) | PyTorch, Python | `nn.Module`, Tensor Operations, Backpropagation, SGD / Adam |
| **04** | [`Lab-04-Keras-MLP-Multiclass-Classification`](./Lab-04-Keras-MLP-Multiclass-Classification) | Multiclass Flower Classification | Keras, TensorFlow, Scikit-learn | Multi-Layer Perceptron (MLP), Softmax, Categorical Cross-Entropy |
| **05** | [`Lab-05-CNN-Binary-Classification-CIFAR10`](./Lab-05-CNN-Binary-Classification-CIFAR10) | Binary Image Classification (Cats vs Dogs) | TensorFlow, Keras | CNN Architecture, Conv2D, MaxPooling2D, Flatten, Dense, Dropout |
| **06** | [`Lab-06-Keras-MLP-Regression`](./Lab-06-Keras-MLP-Regression) | Deep MLP Regression Analysis | Keras, TensorFlow | Activation Functions (ReLU, Sigmoid, Tanh), Loss Analysis (MSE, MAE, Huber) |
| **07** | [`Lab-07-Keras-MLP-Multiclass-Classification`](./Lab-07-Keras-MLP-Multiclass-Classification) | Multiclass Wine Cultivar Classification | Keras, TensorFlow, Scikit-learn | Multi-Layer Perceptron (MLP), Softmax, Categorical Crossentropy, Activation/Optimizer Benchmarks |

---

## 📂 Repository Directory Structure

```text
Artificial-Neural-Networks-and-Deep-Learning-Labs/
├── README.md                                          # Master documentation & syllabus index
├── .gitignore                                         # Git ignore rules for ML/DL artifacts
│
├── Lab-01-Perceptron-Irrigation-Decision/
│   ├── Lab-1.py                                       # Standalone Python script
│   ├── Lab_1.ipynb                                    # Jupyter Notebook walkthrough
│   └── README.md                                      # Lab 1 problem statement & math derivation
│
├── Lab-02-Single-Neuron-Perceptron-Loan-Approval/
│   ├── lab-2.py                                       # NumPy Perceptron implementation
│   ├── perceptron_loan_approval.ipynb                 # Interactive notebook
│   └── README.md                                      # Decision boundary explanation
│
├── Lab-03-Feedforward-NN-PyTorch/
│   ├── lfeedforward_nn.py                             # PyTorch nn.Module architecture
│   ├── Lab_3.ipynb                                    # Training loops and loss progression
│   └── README.md                                      # Feedforward theory & PyTorch tensors
│
├── Lab-04-Keras-MLP-Multiclass-Classification/
│   ├── iris_classifier.py                             # Keras sequential MLP pipeline
│   └── README.md                                      # One-hot encoding & Softmax evaluation
│
├── Lab-05-CNN-Binary-Classification-CIFAR10/
│   ├── Lab_5_CNN_Cats_vs_Dogs.ipynb                   # End-to-end CNN pipeline on CIFAR-10
│   └── README.md                                      # Feature maps, pooling & filters
│
├── Lab-06-Keras-MLP-Regression/
│   ├── Lab_6_Keras_MLP_Regression.ipynb               # Benchmark of activations & loss metrics
│   └── README.md                                      # Regression diagnostics & convergence
│
└── Lab-07-Keras-MLP-Multiclass-Classification/
    ├── Lab_7_Keras_MLP_Multiclass_Classification.ipynb# Interactive Google Colab / Jupyter notebook
    ├── keras_mlp_multiclass.py                         # Standalone Python training pipeline
    ├── lab-7.py                                        # Execution entrypoint
    ├── README.md                                       # Full academic report & 13 question analysis
    └── plots/                                          # Generated confusion matrices & learning curves
```

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vaibhav007-star/Artificial-Neural-Networks-and-Deep-Learning-Labs.git
cd Artificial-Neural-Networks-and-Deep-Learning-Labs
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision tensorflow jupyter
```

### 4. Running a Specific Lab
To run any lab's Python script:
```bash
python Lab-01-Perceptron-Irrigation-Decision/Lab-1.py
```
Or launch Jupyter to explore the notebooks interactively:
```bash
jupyter notebook
```

---

## 👤 Author
**Vaibhav**
- GitHub: [@Vaibhav007-star](https://github.com/Vaibhav007-star)
- Specialization: AI & Data Science Engineering • Multi-Agent Systems • Deep Learning
