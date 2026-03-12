import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    result = F.softmax((Q @ K.transpose(1, 2)) / math.sqrt(Q.size(-1)), dim = -1) @ V
    return result