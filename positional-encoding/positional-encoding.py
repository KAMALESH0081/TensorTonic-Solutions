import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):

    position = np.arange(seq_len, dtype=float)[:, np.newaxis]

    dim = np.arange(d_model, dtype=float)[np.newaxis, :]

    pair_index = np.floor(dim / 2)

    angle_rates = base ** (-2 * pair_index / d_model)

    angles = position * angle_rates

    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe