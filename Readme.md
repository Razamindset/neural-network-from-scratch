# Neural Network from Scratch

This project contains several implementations of neural networks built from scratch using Python and NumPy. It's a collection of different approaches to understanding and building neural networks, from a simple single-layer perceptron to a more complete neural network class. I after some experince in programming have come to understand that if u have time build things from scratch. The understanding u get this way is unprecedented.

## Project Structure

The project is divided into three main parts, each in its own directory under `src/`:

- `single-layer-perceptron`: A basic implementation of a single-layer perceptron that learns the AND gate logic.
- `nn-from-blog`: A neural network implementation based on the concepts from Victor Zhou's blog post, "A simple neural network from scratch." This includes detailed comments and explanations.
- `neural-network`: A more generalized and reusable `NeuralNetwork` class that can be configured with different input, hidden, and output layer sizes.(In Work)

## Implementations

### 1. Single-Layer Perceptron

- **File:** `src/single-layer-perceptron/main.py`
- **Description:** This is the simplest form of a neural network. It's a single-layer perceptron that learns to mimic the behavior of a logical AND gate. It uses a step function for activation and demonstrates the basic principles of weight and bias updates.

### 2. Neural Network from a Blog Post

- **File:** `src/nn-from-blog/main.py`
- **Description:** This implementation is based on the tutorial by Victor Zhou. It's a simple neural network with two inputs, a hidden layer with two neurons, and an output layer with one neuron. The code is heavily commented to provide a clear understanding of the concepts.

### 3. General Neural Network Class

- **File:** `src/neural-network/nn.py`
- **Description:** This is a more flexible and reusable implementation of a neural network. It's a class that can be instantiated with a specified number of input, hidden, and output neurons. It uses the sigmoid activation function and backpropagation to learn from the data. The example usage in the file demonstrates how to train the network on the XOR problem.

## Getting Started

### Prerequisites

- Python 3
- NumPy

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/neural-network-from-scratch.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

You can run each implementation separately:

- **Single-Layer Perceptron:**
  ```bash
  python src/single-layer-perceptron/main.py
  ```
- **Neural Network from Blog:**
  ```bash
  python src/nn-from-blog/main.py
  ```
- **General Neural Network:**
  ```bash
  python src/neural-network/nn.py
  ```

## Attributions

- The `nn-from-blog` implementation is based on the blog post by Victor Zhou: [A simple neural network from scratch](https://victorzhou.com/blog/intro-to-neural-networks/)
- The `single-layer-perceptron` is inspired by this YouTube video: [Single Layer Perceptron AND Gate](https://www.youtube.com/watch?v=OFbnpY_k7js)
