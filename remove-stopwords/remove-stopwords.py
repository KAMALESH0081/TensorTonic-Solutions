def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    hash_table = {ele : ele for ele in stopwords}

    result = [token for token in tokens if token not in hash_table]

    return result