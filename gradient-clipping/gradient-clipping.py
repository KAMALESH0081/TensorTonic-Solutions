import numpy as np

def clip_gradients(g, max_norm):
    g_arr = np.asarray(g)
    
    # Edge case: non-positive max_norm
    if max_norm <= 0:
        return g_arr.copy()
    
    grad_norm = np.linalg.norm(g_arr)
    
    # Edge case: zero norm
    if grad_norm == 0:
        return g_arr.copy()
    
    if grad_norm <= max_norm:
        return g_arr.copy()
    
    scale = max_norm / grad_norm
    return g_arr * scale
       