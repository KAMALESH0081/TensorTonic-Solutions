import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    
    if rng is not None:
        random_matrix = rng.random(x.shape)
    else:
        random_matrix = np.random.random(x.shape)
    
    deactivation = np.where(random_matrix < (1-p) , x, 0)
    drpout_patrn_wth_no_scale = np.where(deactivation == 0, 0, 1)

    scaling_norm = 1 / (1 - p)
    
    output = deactivation * scaling_norm
    dropout_pattern = drpout_patrn_wth_no_scale * scaling_norm

    return (output, dropout_pattern)