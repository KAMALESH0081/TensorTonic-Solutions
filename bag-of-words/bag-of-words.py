import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vocab_dict = {}
    for token in vocab:
        vocab_dict[token] = 0

    for token in tokens:
        if token in vocab_dict:
            vocab_dict[token] += 1

    result = np.array(list(vocab_dict.values()), dtype = int)
    return result