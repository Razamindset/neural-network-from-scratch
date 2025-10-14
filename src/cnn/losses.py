import numpy as np

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

#! When this code was written only chatgpt and god knew how it worked. 
#! Stil only god and chatgpt know, how this work
def categorical_cross_entropy(y_true, y_pred):
    # Small epsilon to prevent log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

def categorical_cross_entropy_prime(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true / y_pred
