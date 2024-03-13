import numpy as np


def diff_embeddings(a, b, dist='cosine') -> float:
    if dist == 'cosine':
        return 1.0 - np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        return np.linalg.norm(a - b)
