import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    result = []
    for i in range(len(y_true)):
        p_h = max(eps, (min(y_pred[i], 1 - eps)))
        log_loss = -((y_true[i] * math.log(p_h)) + ((1 - y_true[i]) * math.log(1 - p_h)))
        result.append(log_loss)

    return result