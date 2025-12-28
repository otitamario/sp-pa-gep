import numpy as np

def resolvent_linear_identity(x, r):
    """
    Resolvent for f(x,y) = <x, y-x> in R^n.
    Closed form: S_r(x) = x / (1 + r)
    """
    return x / (1.0 + r)
