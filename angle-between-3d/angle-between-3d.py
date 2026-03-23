import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.asarray(v)
    w = np.asarray(w)

    num = v @ w.T

    den = np.linalg.norm(v) * np.linalg.norm(w)

    cos_theta = num / den

    theta = np.arccos(cos_theta)

    return theta