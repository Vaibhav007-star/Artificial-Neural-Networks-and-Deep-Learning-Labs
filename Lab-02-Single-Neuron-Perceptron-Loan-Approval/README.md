# Single Neuron Perceptron — Loan Approval Example

A minimal, from-scratch implementation of a **single neuron perceptron** (using only NumPy) applied to a simple real-life binary decision problem: **loan approval**.

## Problem Statement

A bank approves a loan only if **both** conditions are satisfied:
1. Applicant has a **good credit score**
2. Applicant has a **stable income**

This is logically equivalent to an **AND gate** — a classic example used to demonstrate how a single perceptron can learn linearly separable functions.

| Good Credit | Stable Income | Loan Approved |
|:-----------:|:-------------:|:-------------:|
| 0           | 0             | 0 (Rejected)  |
| 0           | 1             | 0 (Rejected)  |
| 1           | 0             | 0 (Rejected)  |
| 1           | 1             | 1 (Approved)  |

## How It Works

The perceptron learns the decision rule using the classic perceptron learning algorithm:

1. **Weighted sum**: `linear_output = dot(inputs, weights) + bias`
2. **Activation (step function)**: outputs `1` if `linear_output >= 0`, else `0`
3. **Error**: `error = actual_label - prediction`
4. **Weight update**: `weights += learning_rate * error * inputs`
5. **Bias update**: `bias += learning_rate * error`

Repeated over multiple epochs, the weights converge so the perceptron correctly separates the "approved" case from the "rejected" cases.

## Files

- `perceptron_loan_approval.ipynb` — Jupyter notebook with step-by-step explanation, training, and testing
- `code.py` — original plain Python script (AND gate perceptron)

## Requirements

```bash
pip install numpy
```

## Usage

Open the notebook in Jupyter:

```bash
jupyter notebook perceptron_loan_approval.ipynb
```

Or run the script directly:

```bash
python code.py
```

## Example Output

```
Good Credit, Stable Income -> Loan APPROVED
Good Credit, Unstable Income -> Loan REJECTED
Bad Credit, Stable Income -> Loan REJECTED
Bad Credit, Unstable Income -> Loan REJECTED
```

## Why This Matters

This simple example demonstrates the foundation of neural networks: a single artificial neuron learning a decision boundary from labeled data. The same "both conditions must be true" pattern generalizes to real systems like:
- Access control (password AND device verified)
- Eligibility checks (age requirement AND documentation complete)
- Basic fraud flags (unusual location AND unusual amount)

## License

Free to use for learning and educational purposes.
