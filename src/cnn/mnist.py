import pandas as pd
import numpy as np
from dense import Dense
from activations import Sigmoid, Softmax
from network import train, predict
from losses import mse, mse_prime

train_data = pd.read_csv("datasets/mnist_train.csv")
test = pd.read_csv("datasets/mnist_test.csv")

X_train = np.array(train_data.iloc[:, 1:]) / 255.0
y_train = np.array(train_data.iloc[:, 0])

X_test = np.array(test.iloc[:, 1:]) / 255.0
y_test = np.array(test.iloc[:, 0])

print(f"{X_train.shape}, {y_train.shape}")
print(f"{X_test.shape}, {y_test.shape}")

# Add convolution and dense layers
network = [
    # Input size, output size
    Dense(784, 128),
    Sigmoid(),
    Dense(128, 10),
    Softmax(),
]

train(network, mse , mse_prime, X_train, y_train, epochs=1000, learning_rate=0.01, verbose=True)

for x, y in zip(X_test, y_test):
    prediction = predict(network, x)
    print(f"Prediction: {prediction}, Actual: {y}")





