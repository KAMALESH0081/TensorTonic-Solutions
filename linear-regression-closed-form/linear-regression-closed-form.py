import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X)
    y = np.array(y)
    
    inter1 = np.linalg.inv(X.T @ X)
    inter2 = X.T @ y

    result = inter1 @ inter2
    return result