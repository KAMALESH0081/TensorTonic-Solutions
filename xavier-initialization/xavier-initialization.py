import math

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    L = math.sqrt(6 / (fan_in + fan_out))

    result = []

    for i in range(len(W)):
        node = []
        for j in range(len(W[0])):
            node.append((W[i][j] * 2 * L) - L)

        result.append(node)

    return result