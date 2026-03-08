import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    vocab_index = {word: i for i, word in enumerate(vocab)}

    result = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in vocab_index:
            result[vocab_index[token]] += 1

    return result