import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        """Initilize the weights and biases for this layer"""
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size, 1)
    
    def forward(self, X):
        # Reshaping to (1, n_features)
        self.input = X.reshape(1, -1)
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        # Reshaping it to 2d (1, n_gradients) 
        output_gradient = output_gradient.reshape(1, -1)

        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.biases += learning_rate * output_gradient
        return input_gradient