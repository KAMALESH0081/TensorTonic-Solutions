def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(W[0])):
            element = 0
            for k in range(len(W)):
                element += (X[i][k] * W[k][j])
            row.append(element + b[j])
        result.append(row)
    return result