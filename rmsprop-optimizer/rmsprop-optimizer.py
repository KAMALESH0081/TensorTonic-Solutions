import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.array(w)
    s = np.array(s)
    g = np.array(g)
    
    s_t = (s * beta) + (1 - beta) * g ** 2
    w = w - (lr/(np.sqrt(s_t + 1e-8))) * g
    
    return (w, s_t)