def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    result = {}

    for sentence in sentences:
        for word in sentence:
            if word in result:
                result[word] += 1
            else:
                result[word] = 1

    return result