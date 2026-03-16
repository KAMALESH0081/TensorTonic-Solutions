import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)

    if X.shape[0] < 2 or X.ndim < 2:
        return None
        
    N = X.shape[0]

    mean = np.mean(X, axis = 0, keepdims=True)
    
    X = X - mean

    cov = (X.T @ X) / (N - 1)
    
    return cov
    