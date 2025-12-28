"""
Algorithms: SPPA, WPPA, (later PPA, EG)
"""
import numpy as np
from src.utils import stopping_criterion

def SPPA(x0, u, alpha_seq, r, resolvent, tol=1e-6, max_iter=10_000):
    """
    Strong Proximal Point Algorithm (Algorithm 4.1)
    """
    x = x0.copy()
    history = [x.copy()]

    for k, alpha in enumerate(alpha_seq):
        u_k = resolvent(x, r)
        x_new = alpha * u + (1.0 - alpha) * u_k
        history.append(x_new.copy())

        if stopping_criterion(x_new, x, tol) or k >= max_iter:
            break

        x = x_new

    return np.array(history)


def WPPA(x0, alpha_seq, r, resolvent, tol=1e-6, max_iter=10_000):
    """
    Weak Proximal Point Algorithm (Algorithm 4.2)
    """
    x = x0.copy()
    history = [x.copy()]

    for k, alpha in enumerate(alpha_seq):
        u_k = resolvent(x, r)
        x_new = alpha * x + (1.0 - alpha) * u_k
        history.append(x_new.copy())

        if stopping_criterion(x_new, x, tol) or k >= max_iter:
            break

        x = x_new

    return np.array(history)
