import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x)
    # Calculate sigmoid element-wise
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Use * for element-wise multiplication
    res = x * sigmoid
    
    return np.asarray(res, dtype=float)