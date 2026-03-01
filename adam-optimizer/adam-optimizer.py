import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * grad ** 2

    m_c = m_t / (1 - beta1 ** t)
    v_c = v_t / (1 - beta2 ** t)

    param_new = param - lr * (m_c / (np.sqrt(v_c) + eps))

    return param_new, m_t, v_t