import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
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

    

    