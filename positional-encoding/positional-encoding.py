import numpy as np

def dim_list(d_model):
    dlist = []
    for i in range(0, d_model, 2):
        dlist.append(i)
        dlist.append(i)
    return dlist

def positional_encoding(seq_len, d_model, base=10000.0):
    j = dim_list(d_model)
    m = np.zeros((seq_len, d_model))
    for t in range(seq_len):
        for k in range(d_model):
            if k % 2 == 0:
               m[t][k] = np.sin(t/(base ** ((j[k])/d_model)))
            else:
               m[t][k] = np.cos(t/(base ** ((j[k])/d_model)))
    return m