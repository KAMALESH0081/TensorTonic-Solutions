import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    
        seq_len = token_ids.shape[1]
    
        final_token_embeddings = self.token_embeddings[token_ids]
    
        final_position_embeddings = self.position_embeddings[:seq_len]
    
        final_segment_embeddings = self.segment_embeddings[segment_ids]
    
        final_embeddings = final_token_embeddings + final_position_embeddings + final_segment_embeddings
    
        return final_embeddings
        

        
        
        
