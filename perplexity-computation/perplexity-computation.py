import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """

    sum_log_probs = 0
    N = len(actual_tokens)
    
    for i in range(N):
        sum_log_probs += math.log(prob_distributions[i][actual_tokens[i]])

    result = math.exp(-1 * (sum_log_probs/ N))

    return result