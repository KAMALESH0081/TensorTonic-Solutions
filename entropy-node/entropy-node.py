import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    counts = np.unique(y, return_counts = True)
    ratio = counts[1] / len(y)
    entropy = -1 * (np.sum(ratio * np.log2(ratio)))
    return entropy