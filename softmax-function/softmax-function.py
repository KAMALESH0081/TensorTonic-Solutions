import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x)
    x_new = x - np.max(x, axis = x.ndim - 1, keepdims = True)
    x_exp = np.exp(x_new)
    return x_exp  / np.sum(x_exp, axis = x.ndim - 1, keepdims = True)
    