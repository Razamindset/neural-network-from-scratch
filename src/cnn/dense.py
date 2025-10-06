import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        """Initilize the weights and biases for this layer"""
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random(output_size, 1)
    
    def forward(self, X):
        self.input = X
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= self.learning_rate * weights_gradient
        self.biases += self.learning_rate * output_gradient
        return input_gradient