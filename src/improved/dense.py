import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        """Initilize the weights and biases for this layer"""
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(1, output_size)
    
    def forward(self, X):
        # Reshaping to (1, n_features)
        self.input = X.reshape(1, -1)

        # print(f"[DEBUG] Input: {self.input}, Weights: {self.weights}")

        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):

        output_gradient = output_gradient.reshape(1, -1)

        weights_gradient = np.dot(self.input.T, output_gradient)

        # print(f"[DEBUG] Input Shape: {self.input.shape}")
        # print(f"[DEBUG] Gradient Shape: {output_gradient.shape}")

        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient