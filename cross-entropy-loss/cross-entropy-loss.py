import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    
    p_x_y_array = y_pred[np.arange(y_true.shape[0]), y_true]

    result = -1 * (np.mean(np.log(p_x_y_array)))

    return result