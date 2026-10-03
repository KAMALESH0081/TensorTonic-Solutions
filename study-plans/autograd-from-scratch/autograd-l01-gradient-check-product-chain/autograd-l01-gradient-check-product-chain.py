import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    def forward(a, b, c, f):
        return (a * b + c) * f
    e = (a * b + c)
    loss = e * f
    a_d_a = b * f
    a_d_b = a * f
    a_d_c = f
    a_d_f = e

    base = forward(a, b, c, f)
    n_a = (forward(a + h, b, c, f) - base) / h
    n_b = (forward(a, b + h, c, f) - base) / h
    n_c = (forward(a, b, c + h, f) - base) / h
    n_f = (forward(a, b, c, f + h) - base) / h

    abs_diff = max(abs(a_d_a - n_a), abs(a_d_b - n_b), abs(a_d_c - n_c), abs(a_d_f - n_f))
    
    return (float(loss),
            [float(a_d_a), float(a_d_b), float(a_d_c), float(a_d_f)],
            [float(n_a), float(n_b), float(n_c), float(n_f)], float(abs_diff))