import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes a simple feedforward neural network.

        Args:
            input_size (int): Number of features in the input layer.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer.
        """
        self.w1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        pass
    
    def sigmoid(self, X):
        """f(x) = 1 / (1 + e^(-x))"""
        return 1/(1+ np.exp(-X))

    def sigmoid_derivitive(self, x):
        """This helps when back propogating"""
        s = self.sigmoid(x)
        return s * (1 - s)    
    
    
    def forward(self, X):
        """Input -> Hidden -> Output"""
        # * Numpy allows us to process and calculate all the nodes in one line
        
        # z1 will have weighted sums for each of the nodes in the hidden layer
        # a1 will contain activation for each of the node
        # z1 and a1 have sizes wrt hidden layer (1, hidden_size)
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # We do the same for the last layer take all the activation and process them
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2