import numpy as np

def sigmoid(X):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    """
    Derivative of the sigmoid function. This is used during backpropagation.
    d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    s = sigmoid(X)
    return s * (1 - s)    

def mse(y_true, y_pred):
    """MSE: Mean Squared Error"""
    return np.mean(np.square(y_true - y_pred))

def mse_prime(y_true, y_pred):
    """Derivative of the mean squared Error"""
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)