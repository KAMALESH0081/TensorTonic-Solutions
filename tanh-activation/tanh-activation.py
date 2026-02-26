import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)
    num = np.exp(x) - np.exp(-x)
    den = np.exp(x) + np.exp(-x)
    return num / den