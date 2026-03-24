def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    dot_product = sum(a * b for a, b in zip(x1, x2))

    x1_mag = math.sqrt(sum(x**2 for x in x1))
    x2_mag = math.sqrt(sum(x**2 for x in x2))

    cos = dot_product / (x1_mag * x2_mag)

    if label == 1:
        return 1 - cos
    else:
        return max(0, cos - margin)