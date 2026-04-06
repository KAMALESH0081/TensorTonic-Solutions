import math

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []
    
    for i in x:
        if i > 0:
            result.append(i)
        else:
            result.append(alpha * (math.exp(i) - 1))
            
    return result
    