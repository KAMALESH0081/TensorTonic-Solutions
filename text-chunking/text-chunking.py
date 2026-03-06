def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    result = []
    if tokens == []:
        return result
    moving_idx = 0
    
    while len(tokens[moving_idx:]) > chunk_size:
        result.append(tokens[moving_idx:moving_idx + chunk_size])
        moving_idx = moving_idx + chunk_size - overlap
    else:
        result.append(tokens[moving_idx:len(tokens)])
    return result
        
        
    
        