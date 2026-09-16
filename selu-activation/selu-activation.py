import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    scale_factor = 1.0507
    alpha = 1.6733
    result = []
    for x in x:
        if x > 0:
            result.append(scale_factor * x)
        else:
            result.append(scale_factor * alpha * (math.exp(x) - 1))

    return result