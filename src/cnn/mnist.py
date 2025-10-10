import pandas as pd
import numpy as np

train = pd.read_csv("datasets/mnist_train.csv")
test = pd.read_csv("datasets/mnist_test.csv")

X_train = np.array(train.iloc[:, 1:])
y_train = np.array(train.iloc[:, 0])

X_test = np.array(test.iloc[:, 1:])
y_test = np.array(test.iloc[:, 0])

print(f"{X_train.shape}, {y_train.shape}")
print(f"{X_test.shape}, {y_test.shape}")
