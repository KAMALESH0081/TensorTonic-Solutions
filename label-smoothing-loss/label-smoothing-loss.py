import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)
    epsilon_by_k = epsilon / K
    smoothed_targets = [epsilon_by_k] * K

    smoothed_targets[target] = (1 - epsilon) + epsilon_by_k

    negative_sum_probs = 0

    for i in range(K):
        negative_sum_probs += smoothed_targets[i] * math.log(predictions[i])

    return -1 * negative_sum_probs

    