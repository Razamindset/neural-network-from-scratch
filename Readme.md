# Neural Network from Scratch

This repository is a **from-scratch implementation of neural networks using Python and NumPy**, built with the goal of *deep conceptual understanding rather than convenience*.
After gaining some experience in programming, I realized that if you truly want to understand how things work, **building them from scratch is unmatched**. This project reflects that philosophy.

The code evolves from basic perceptrons to a **modular, extensible neural network framework**, supporting dense layers, activations, losses, training loops, and even MNIST classification — all without using deep learning libraries like PyTorch or TensorFlow.

---

## Project Structure

```
neural-network-from-scratch/
│
├── datasets/
│   └── (datasets such as MNIST)
│
├── src/
│   ├── improved/
│   │   ├── activation.py        # Base activation class
│   │   ├── activations.py       # ReLU, Sigmoid, Softmax, etc.
│   │   ├── convolution.py       # (Experimental / WIP)
│   │   ├── dense.py             # Fully connected (Dense) layer
│   │   ├── losses.py            # Loss functions (MSE, Cross-Entropy)
│   │   ├── mnist.py             # MNIST dataset loading & testing
│   │   ├── network.py           # Neural network orchestration
│   │   ├── reshape.py           # Shape transformation layer
│   │   └── train.py             # Training loop & utilities
│   │
│   ├── neural-network-with-numpy/
│   │   └── nn.py                # Earlier Simple fixed 3 layers implementation
│   │
│   └── single-layer-perceptron/
│       ├── main.py              # AND-gate perceptron
│       ├── design_log.md
│       ├── activation.png
│       └── perceptron.png
│
├── README.md
├── requirements.txt
└── venv/
```

---

## Implementations Overview

### 1. Single-Layer Perceptron

* **Path:** `src/single-layer-perceptron/main.py`
* **Description:**
  A minimal implementation of a single-layer perceptron trained to learn the **AND gate**.
  This part focuses on:

  * Step activation function
  * Weight & bias updates
  * Linear separability

This serves as the **conceptual foundation** of neural networks.

---

### 2. Early Neural Network (NumPy-Based)

* **Path:** `src/neural-network-with-numpy/nn.py`
* **Description:**
  An earlier implementation of a small neural network using NumPy arrays, sigmoid activation, and manual backpropagation.
  It demonstrates:

  * Forward propagation
  * Backpropagation
  * Training on simple problems like XOR

This version helped bridge the gap between perceptrons and fully modular architectures.

---

### 3. Modular Neural Network Framework (Current / Main)

* **Path:** `src/improved/`
* **Status:** Actively evolving 🚧

This is the **core of the project** — a modular neural network system inspired by real deep-learning frameworks, but implemented entirely from scratch.

#### Key Components

* **Layers**

  * `Dense`: Fully connected layer
  * `Reshape`: Shape manipulation layer
  * `Convolution`: Experimental (WIP)

* **Activations**

  * Sigmoid
  * ReLU
  * Softmax
  * Custom activation base class

* **Loss Functions**

  * Mean Squared Error (MSE)
  * Cross-Entropy Loss

* **Network Engine**

  * Forward pass chaining
  * Backward pass with gradients
  * Weight updates
  * Training loop abstraction

* **Datasets**

  * MNIST loading and evaluation

This structure mirrors how real frameworks work internally, but keeps everything **explicit and readable**.

---

## Getting Started

### Prerequisites

* Python 3.8+
* NumPy

---

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Code

### Single-Layer Perceptron

```bash
python src/single-layer-perceptron/main.py
```

### Early NumPy Neural Network

```bash
python src/neural-network-with-numpy/nn.py
```

### Modular Neural Network (MNIST / Experiments)

```bash
python src/improved/train.py
```

---

## Design Philosophy

* ❌ No PyTorch / TensorFlow
* ✅ Explicit math and gradients
* ✅ Readability over abstraction
* ✅ Learning-first, performance-second

This project is meant to **teach**, not hide details behind APIs.

---

## Future Work

* Complete convolution layer implementation
* Add optimizers (Momentum, Adam)
* Batch normalization
* Model saving/loading
* Performance improvements
* NNUE-style optimizations (for future chess-engine integration)

---

## Attributions

* Inspired by Victor Zhou’s article:
  **“A Simple Neural Network from Scratch”**
  [https://victorzhou.com/blog/intro-to-neural-networks/](https://victorzhou.com/blog/intro-to-neural-networks/)

---
