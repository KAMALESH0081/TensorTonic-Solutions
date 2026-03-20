import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    x_new = x - np.max(x, axis = x.ndim - 1, keepdims = True)
    
    return np.exp(x_new)  / np.sum(np.exp(x_new), axis = x.ndim - 1, keepdims = True)
    