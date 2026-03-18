import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    result = np.zeros(k + 1)

    for i in range(k + 1):
        r = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        result[i] = r

    return (result[-1].item(), (np.sum(result).item()))