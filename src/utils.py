"""
Utilities
"""
import numpy as np

def norm(x):
    return np.linalg.norm(x)

def stopping_criterion(x_new, x_old, tol=1e-6):
    return norm(x_new - x_old) <= tol
