import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.array(A)
    result = 0
    for i in range(len(A)):
        result += A[i, i]

    return result