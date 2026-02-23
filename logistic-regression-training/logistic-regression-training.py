import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape
    w = np.zeros((D))
    b = 0.0
    
    for _ in range(steps):
        z = (X @ w) + b
        y_h = _sigmoid(z)
        g_s = y_h - y
        g_w = (X.T @ g_s)/ N
        g_b = np.mean(g_s)
        w = w - lr * g_w
        b = b - lr * g_b

    return (w, b)
        
        
        
 
