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
        self.learning_rate = 0.1
    
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
    
    def  backward(self, X, y_true, output_pred):
        pass
    
    
    def calculate_loss(self, y_true, y_pred):
        # Mean Squared Error (MSE)
        y_true_reshaped = y_true.reshape(y_pred)
        return np.mean(np.square(y_true_reshaped - y_pred))
    
    def train(self, X_train, y_train, epochs):
        print(f"Training started with learning rate: {self.learning_rate}")
        for epoch in range(epochs):
            out_pred = self.forward(X_train)

            # Calculate Loss using mean squared.. error = actaul - predicted -> error_squared
            loss = self.calculate_loss(y_train, out_pred)
            
            # Back propgate to update the weights and biases
            self.backward(X_train, y_train, out_pred)
            
            # Print epochs every 10th of the total. If total is < 100 then default to 10 epochs 
            if (epoch + 1) % (epochs // 10 if epochs >= 100 else 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        print("Training Finished")
            