import numpy as np

def dot_product(x, y):
    x = np.array(x)
    y = np.array(y)
    return x @ y.T