import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    
    if np.sum(p) <= 1 + 1e-6 and np.sum(p) >= 1 - 1e-6 and len(x) == len(p):
        return np.sum(x * p)
    else:
        raise ValueError("ValueError")



    