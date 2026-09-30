import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    l = len(coefficients)
    f_x_h = 0
    f_x = 0
    for i in range(1, l + 1):
        f_x_h += coefficients[-i] * ((x+h)**(l-i))
        f_x += coefficients[-i] * (x**(l-i))
    final = (f_x_h - f_x) / h
    return (float(f_x), float(f_x_h), float(final))

        
        
        
