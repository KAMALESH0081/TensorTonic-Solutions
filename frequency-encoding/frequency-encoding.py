def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    look_up = {}
    for val in values:
        if val not in look_up:
            look_up[val] = 1
        else:
            look_up[val] += 1
    n = len(values)    
    result = [int(look_up.get(val, val))/n for val in values]
    return result