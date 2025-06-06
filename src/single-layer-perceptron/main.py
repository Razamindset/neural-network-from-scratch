import numpy as np

# This single layer perceptron will train for (And gate)^ logic
# A perceptron only produces binary output

# Inputs: Take each row as a training example
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Expected Outputs
y = np.array([0, 0, 0, 1])

# Weights and biases
weights = np.random.rand(2) 
bias = np.random.rand()

# Learning rate
lr = 0.1


# Step activation function
def step(x):
    return 1 if x > 0 else 0

for epoch in range(10):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        # lets do the wighted sum
        # X[i] -> [0, 1] (One training example)
        # z = [0, 1] dot [w1, w2] (1D vector)
        z = np.dot(X[i], weights) + bias 
        # The above line is equal to : z = x1.w1 + x2.w2 + bias
        
        y_pred = step(z)
        
        error = y[i] - y_pred
        
        # Update weights and bias to reduce prediction error:
        # If the prediction is incorrect, adjust weights in the direction of the input.
        # This helps the model better classify this input in the future.
        weights += lr * error * X[i]
        
        bias += lr* error
        
        print(f"Input: {X[i]} | Predicted: {y_pred} | Error: {error}") 

print("\nTesting trained perceptron:")
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    y_pred = step(z)
    print(f"Input: {X[i]} → Predicted: {y_pred} | Actual: {y[i]}")
