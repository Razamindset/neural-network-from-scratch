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
class Softmax:
    # When using softmax prime with cross entropy loss we donot need
    # softmax prime the gradient of the ouput containing the calc for 
    # softmax prime can be represented as loss = predicted - target 
    def __init__(self):
        pass

    def forward(self, input):
        """For each comp of z-> activation=(e^z)/sum(e^z1 + e^z2 + ...)"""
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    # I dont understand shit about this. 
    # Just feels okay
    # We need to look at this in free time
    # See page number 5 for mathematics
    # The backdrop seems do some magic and make the gradients layer for prev layer
    # Where these can be used to calculate weights gradients
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

