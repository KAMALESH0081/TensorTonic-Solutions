import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x, dtype = float)
    
    swish = x * (1/ (1 + np.exp(-x)))

    return swish