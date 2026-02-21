import numpy as np

def clip_gradients(g, max_norm):
    g_1 = np.array(g)
        # Edge case: non-positive max_norm
    if max_norm <= 0:
        return g_1
    
    grad_norm = np.linalg.norm(g_1)

    if grad_norm > 0 and grad_norm > max_norm:
        return g_1*(max_norm/grad_norm)
    else:
        return g_1
       