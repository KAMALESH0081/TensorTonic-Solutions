import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    T = len(v)
    result = np.zeros((T, T))

    for i in range(T):
        result[i, i] = v[i]

    return result
