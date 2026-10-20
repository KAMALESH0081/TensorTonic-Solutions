import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)

    new_w = values - (gradients * learning_rate)
    loc_pred = np.sum(gradients * (new_w - values))
    
    return (new_w.tolist(), loc_pred.tolist())
