import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)

    if x.ndim == 2:
        axis = 0
        gamma = gamma.reshape(1, -1)
        beta = beta.reshape(1, -1)
        
    else:
        axis = (0, 2, 3)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

    mean = np.mean(x, axis = axis, keepdims = True)
    var = np.var(x, axis = axis, keepdims = True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    
    result = gamma * x_norm + beta
    
    return gamma * x_norm + beta