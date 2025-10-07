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
        
        # --- Step 1 is to calculate the error over the output layer ---
        # This is the derivative of the loss with respect to the output layer's pre-activation (z2).
        # By the chain rule: d(Loss)/d(z2) = d(Loss)/d(a2) * d(a2)/d(z2)
        # d(Loss)/d(a2) is the derivative of the MSE loss, which is (output_pred - y_true).
        # d(a2)/d(z2) is the derivative of the sigmoid activation function.
        # The result is the 'error' that we'll propagate backward.
        error_at_output_layer = (output_pred - y_true) * self.sigmoid_derivitive(self.z2) # This was 'delta2'
        
        # --- Step 2: Calculate gradients for the output layer weights (w2) and biases (b2) ---
        # The gradient for w2 is d(Loss)/d(w2) = d(Loss)/d(z2) * d(z2)/d(w2)
        # d(z2)/d(w2) is simply the hidden layer's activation, a1.
        # While d(Loss)/d(z2) is from the above calculation
        # We use a dot product to correctly sum the gradients for all samples in the batch.
        gradient_w2 = np.dot(self.a1.T, error_at_output_layer) # This was 'dw2'
        
        #* We use the transpose to align matrix dimensions for the dot product, ensuring the resulting gradient 
        #* matrix has the same shape as the weight matrix we intend to update.
        
        # The gradient for b2 is d(Loss)/d(b2) = d(Loss)/d(z2) * d(z2)/d(b2)
        # d(z2)/d(b2) is 1, so we just sum the error for all samples.
        gradient_b2 = np.sum(error_at_output_layer, axis=0, keepdims=True) # This was 'db2'
        
        # --- Step 3: Propagate the error to the hidden layer ---
        # We calculate the error at the hidden layer's pre-activation (z1).
        # Chain rule: d(Loss)/d(z1) = d(Loss)/d(a2) * d(a2)/d(z2) * d(z2)/d(a1) * d(a1)/d(z1)
        # This simplifies to: [error_at_output_layer .dot. w2.T] * sigmoid_derivative(z1) 
        error_at_hidden_layer = np.dot(error_at_output_layer, self.w2.T) * self.sigmoid_derivitive(self.z1)

        # --- Step 4: Calculate gradients for the hidden layer weights (w1) and biases (b1) ---
        # The gradient for w1 is d(Loss)/d(w1) = d(Loss)/d(z1) * d(z1)/d(w1)
        # d(z1)/d(w1) is the input, X.
        gradient_w1 = np.dot(X.T, error_at_hidden_layer)
        
        # The gradient for b1 is d(Loss)/d(b1) = d(Loss)/d(z1) * d(z1)/d(b1)
        # d(z1)/d(b1) is 1.
        gradient_b1 = np.sum(error_at_hidden_layer, axis=0, keepdims=True)
        
        # --- Step 5: Update all weights and biases ---
      # We adjust the weights and biases in the opposite direction of their gradients.
        self.w2 -= self.learning_rate * gradient_w2
        self.b2 -= self.learning_rate * gradient_b2
        self.w1 -= self.learning_rate * gradient_w1
        self.b1 -= self.learning_rate * gradient_b1
    
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
                