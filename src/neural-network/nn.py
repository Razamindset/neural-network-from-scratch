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
        # Initialize weights with random values
        # w1: weights connecting input layer to hidden layer
        self.w1 = np.random.rand(input_size, hidden_size)
        # b1: biases for the hidden layer
        self.b1 = np.zeros((1, hidden_size))
        
        # w2: weights connecting hidden layer to output layer
        self.w2 = np.random.rand(hidden_size, output_size)
        # b2: biases for the output layer
        self.b2 = np.zeros((1, output_size))
        
        # Learning rate determines the step size during weight updates
        self.learning_rate = 0.1
    
    def sigmoid(self, X):
        """
        Sigmoid activation function.
        f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivitive(self, x):
        """
        Derivative of the sigmoid function. This is used during backpropagation.
        d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        s = self.sigmoid(x)
        return s * (1 - s)    
    
    
    def forward(self, X):
        """
        Performs the forward pass through the neural network.
        Input -> Hidden -> Output

        Args:
            X (np.array): Input data.

        Returns:
            np.array: Predicted output from the network.
        """
        # Calculate weighted sum for the hidden layer (z1)
        # and apply sigmoid activation (a1)
        # z1 and a1 will have shapes (number_of_samples, hidden_size)
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Calculate weighted sum for the output layer (z2)
        # and apply sigmoid activation (a2)
        # z2 and a2 will have shapes (number_of_samples, output_size)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y_true, output_pred):
        """
        Performs the backward pass (backpropagation) to update weights and biases.

        Args:
            X (np.array): Input data used in the forward pass.
            y_true (np.array): True labels corresponding to the input data.
            output_pred (np.array): Predicted output from the forward pass.
        """
        num_samples = X.shape[0]

        # --- Output Layer (Layer 2) Backpropagation ---
        # Calculate the error at the output layer.
        # This is the difference between predicted and true values,
        # multiplied by the derivative of the sigmoid of the output layer's weighted sum.
        # The (output_pred - y_true) gives the gradient of MSE w.r.t. a2.
        # Multiplying by sigmoid_derivative(self.z2) applies the chain rule for the activation.
        # Shape: (num_samples, output_size)
        delta2 = (output_pred - y_true) * self.sigmoid_derivitive(self.z2)
        
        # Calculate the gradient of the weights for w2.
        # This is the transpose of the hidden layer activations multiplied by the output layer error.
        # np.dot(self.a1.T, delta2) ensures the dimensions align correctly:
        # (hidden_size, num_samples) dot (num_samples, output_size) -> (hidden_size, output_size)
        dw2 = np.dot(self.a1.T, delta2)
        
        # Calculate the gradient of the biases for b2.
        # This is the sum of the output layer error along the sample axis (axis=0).
        # np.sum(delta2, axis=0, keepdims=True) ensures the shape remains (1, output_size)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Update weights and biases for the output layer
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        
        # --- Hidden Layer (Layer 1) Backpropagation ---
        # Calculate the error at the hidden layer.
        # This error propagates from the output layer's error (delta2)
        # through the weights connecting the hidden to output layer (self.w2.T),
        # then multiplied by the derivative of the sigmoid of the hidden layer's weighted sum.
        # np.dot(delta2, self.w2.T) ensures dimensions align:
        # (num_samples, output_size) dot (output_size, hidden_size) -> (num_samples, hidden_size)
        # Shape: (num_samples, hidden_size)
        delta1 = np.dot(delta2, self.w2.T) * self.sigmoid_derivitive(self.z1)
        
        # Calculate the gradient of the weights for w1.
        # This is the transpose of the input data multiplied by the hidden layer error.
        # np.dot(X.T, delta1) ensures dimensions align:
        # (input_size, num_samples) dot (num_samples, hidden_size) -> (input_size, hidden_size)
        dw1 = np.dot(X.T, delta1)
        
        # Calculate the gradient of the biases for b1.
        # This is the sum of the hidden layer error along the sample axis (axis=0).
        # np.sum(delta1, axis=0, keepdims=True) ensures the shape remains (1, hidden_size)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Update weights and biases for the hidden layer
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def calculate_loss(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error (MSE) loss.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            float: The mean squared error.
        """
        # Ensure y_true has the same shape as y_pred for element-wise operations
        # No need to reshape y_true if y_pred is already correctly shaped
        # and y_true is also coming in with correct shape from training data
        return np.mean(np.square(y_true - y_pred))
    
    def train(self, X_train, y_train, epochs):
        """
        Trains the neural network using the provided training data.

        Args:
            X_train (np.array): Training input data.
            y_train (np.array): Training true labels.
            epochs (int): Number of training iterations.
        """
        print(f"Training started with learning rate: {self.learning_rate}")
        for epoch in range(epochs):
            # Forward pass: get predictions
            out_pred = self.forward(X_train)

            # Calculate Loss using Mean Squared Error
            loss = self.calculate_loss(y_train, out_pred)
            
            # Backpropagate to update weights and biases
            self.backward(X_train, y_train, out_pred)
            
            # Print loss every 10% of total epochs, or every 10 epochs if total < 100
            print_interval = epochs // 10 if epochs >= 100 else 10
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        print("Training Finished")

# --- Example Usage ---
if __name__ == "__main__":
    # Define input, hidden, and output sizes
    input_size = 2    # e.g., two features like [x, y]
    hidden_size = 4   # number of neurons in the hidden layer
    output_size = 1   # e.g., binary classification (0 or 1)

    # Create a simple dataset for XOR problem (a common test for neural networks)
    # X_train: input features
    # y_train: corresponding true labels
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_train = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ])

    # Initialize the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the network for a certain number of epochs
    epochs = 10000
    nn.train(X_train, y_train, epochs)

    # Test the trained network
    print("\n--- Testing the trained network ---")
    predictions = nn.forward(X_train)
    # Convert predictions to binary (0 or 1) for interpretation
    binary_predictions = (predictions > 0.5).astype(int) 

    print("Input:\n", X_train)
    print("True Output:\n", y_train)
    print("Predicted Output (Raw):\n", predictions)
    print("Predicted Output (Binary):\n", binary_predictions)

    # You can also test with individual inputs
    print("\nTesting individual inputs:")
    test_input1 = np.array([[0, 0]])
    print(f"Input: {test_input1}, Prediction: {(nn.forward(test_input1) > 0.5).astype(int)}")
    
    test_input2 = np.array([[0, 1]])
    print(f"Input: {test_input2}, Prediction: {(nn.forward(test_input2) > 0.5).astype(int)}")
    
    test_input3 = np.array([[1, 0]])
    print(f"Input: {test_input3}, Prediction: {(nn.forward(test_input3) > 0.5).astype(int)}")
    
    test_input4 = np.array([[1, 1]])
    print(f"Input: {test_input4}, Prediction: {(nn.forward(test_input4) > 0.5).astype(int)}")
