import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.array(scores)
    T = scores.shape[-1]
    
    mask_2d = np.tril(np.ones((T, T), dtype=bool))
    result = np.where(mask_2d, scores, mask_value)
    
    return result