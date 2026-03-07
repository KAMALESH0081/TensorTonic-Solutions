import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)

    if x.ndim == 3:
        result = np.mean(x, axis = (1, 2))
    elif x.ndim == 4:
        result = np.mean(x, axis = (2,3))
    else:
        raise ValueError("Value Error")
    return result