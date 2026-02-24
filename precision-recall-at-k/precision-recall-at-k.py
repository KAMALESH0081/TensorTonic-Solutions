def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    hits = 0
    for i in recommended[:k]:
        for j in relevant:
            if i == j:
                   hits += 1

    precision = hits / k
    recall = hits / len(relevant)
    return [precision, recall]
    