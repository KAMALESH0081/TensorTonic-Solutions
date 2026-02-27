import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x)
    condition = x <= 0
    x[condition] = 0
    return x