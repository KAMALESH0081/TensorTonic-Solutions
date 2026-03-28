import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p)
    q = np.asarray(q) + eps

    result = np.sum(np.where(p > 0, (p*(np.log(p/q))), 0))

    return result