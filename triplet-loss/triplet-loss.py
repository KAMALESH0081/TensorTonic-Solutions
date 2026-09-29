import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)

    def dist(x, y):
        return np.sum(np.square(x-y), axis=1)

    L = np.maximum(0.0, (dist(anchor, positive) - dist(anchor, negative) + margin))
    return float(np.mean(L))