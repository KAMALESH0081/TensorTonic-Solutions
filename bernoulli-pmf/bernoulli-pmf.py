import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.array(x, dtype = float)

    mean = p
    var = p * (1 - p)

    result = np.where(x == 0, abs(p-1), p)
    
    return (result, mean, var)
    