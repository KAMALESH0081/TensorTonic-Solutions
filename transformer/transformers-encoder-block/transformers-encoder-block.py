import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalized = (x - mean) / np.sqrt(variance + eps)
    result = (gamma * normalized) + beta

    return result

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    d_k = int(Q.shape[-1] / num_heads)

    #1, 5, 32 @ 1, 32, 32 --> 1, 5, 32
    query = Q @ W_q
    key = K @ W_k
    value = V @ W_v
    
    #1, 5, 32 --> 1, 5, 4, 8 --> 1, 4, 5, 8
    reshaped_q = query.reshape(Q.shape[0], Q.shape[1], num_heads, d_k).transpose(0, 2, 1, 3)
    reshaped_k = key.reshape(K.shape[0], K.shape[1], num_heads, d_k).transpose(0, 2, 1, 3)
    reshaped_v = value.reshape(V.shape[0], V.shape[1], num_heads, d_k).transpose(0, 2, 1, 3)
    
    #1, 4, 5, 8 @ 1, 4, 8, 5 --> 1, 4, 5, 5
    scores = (reshaped_q @ reshaped_k.transpose(0, 1, 3, 2) /  np.sqrt(d_k))
    attention_scores = softmax(scores, axis = -1) 

    #1, 4, 5, 5 @ 1, 4, 5, 8 --> 1, 4, 5, 8
    value_mul = attention_scores @ reshaped_v

    #1, 4, 5, 8 --> 1, 5, 4, 8 --> 1, 5, 32
    concat = value_mul.transpose(0, 2, 1, 3).reshape(Q.shape[0], Q.shape[1], Q.shape[2])

    #1, 5, 32 @ 1, 32, 32 --> 1, 5, 32
    output = concat @ W_o

    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.dot(x, W1) + b1
    
    relu_out = np.maximum(0, hidden)
    
    output = np.dot(relu_out, W2) + b2
    
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x1 = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x2 = x + x1
    x3 = layer_norm(x2, gamma1, beta1)
    x4 = feed_forward(x3, W1, b1, W2, b2)
    x5 = x3 + x4
    x6 = layer_norm(x5, gamma2, beta2)
    return x6















    