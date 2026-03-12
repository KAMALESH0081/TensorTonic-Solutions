import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding_layer = nn.Embedding(vocab_size, d_model)
    return embedding_layer

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    tensor_tokens = embedding(tokens)
    tensor_tokens = tensor_tokens * math.sqrt(d_model)

    return tensor_tokens