import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    L = math.sqrt(6/fan_in)

    result = []
    for col in W:
        column = []
        for element in col:
            column.append((element * 2 * L) - L)
        result.append(column)

    return result