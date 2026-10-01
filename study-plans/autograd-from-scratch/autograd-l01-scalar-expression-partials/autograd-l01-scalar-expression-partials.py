def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    def forward(a, b, c):
        return a * b + c

    def derivative(a, b, c, h, par_val):
        nor_der = forward(a, b, c)
        if par_val == "a":
            a += h
        elif par_val == "b":
            b += h
        else:
            c += h
        par_der = forward(a, b, c)
        return (par_der - nor_der) / h
        
    normal = a * b + c
    par_a = derivative(a, b, c, h, par_val = "a")
    par_b = derivative(a, b, c, h, par_val = "b")
    par_c = derivative(a, b, c, h, par_val = "c")

    return (float(normal), float(par_a), float(par_b), float(par_c))

    