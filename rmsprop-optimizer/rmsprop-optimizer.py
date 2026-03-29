import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w)
    grad = np.asarray(g)
    s = np.asarray(s)

    st = beta * s + (1 - beta) * grad**2
    wt = w - (lr * grad) / (np.sqrt(st)+ eps)
    return wt, st