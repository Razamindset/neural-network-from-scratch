import numpy as np
from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

# This is a different kid of activation we donot need it for early testing
# I donot understand the math completely but i will just use this for now.
class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        exps = np.exp(input - np.max(input))
        self.output = exps / np.sum(exps)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Simple element-wise gradient for softmax + MSE
        return self.output * (output_gradient - np.sum(output_gradient * self.output))
