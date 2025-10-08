import numpy as np
from dense import Dense
from activations import Sigmoid
from network import train, predict
from losses import mse, mse_prime

# Add convolution and dense layers
network = [
    # Input size, output size
    Dense(2, 3),
    Sigmoid(),
    Dense(3, 1),
    Sigmoid(),
]

# And gate inputs for testing
X_train = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
        ])

y_train = np.array([
        1,
        0,
        0,
        0,
        ])

train(network, mse , mse_prime, X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True)

for x, y in zip(X_train, y_train):
    prediction = predict(network, x)
    print(f"Prediction: {prediction}, Actual: {y}")



