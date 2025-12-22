import numpy as np

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input

        output = self.activation(self.input)

        # if output.ndim == 1:
        #     output = output.reshape(1, -1)
        
        return output

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
