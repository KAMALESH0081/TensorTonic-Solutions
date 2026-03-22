import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    
    Counter_list = Counter(x)
    most_freq = Counter_list.most_common(len(Counter_list))
    top_f = most_freq[0][1]
    top_same = [most_freq[0][0]]
    
    for i in range(1, len(most_freq)):
        if most_freq[i][1] < top_f: break
        else:
            top_same.append(most_freq[i][0])
    x = np.array(x)
    return (float(np.mean(x)),float(np.median(x)), float(min(top_same)))

    