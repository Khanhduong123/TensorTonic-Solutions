import numpy as np

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations for f(x) = ax^2 + bx + c.
    """
    # Convert inputs to numpy arrays (fixing the typo)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    x = x0

    for _ in range(steps):
        # Calculate the gradient: d/dx (ax^2 + bx + c) = 2ax + b
        grad = 2 * a * x + b
        
        # Update x
        x = x - lr * grad
        
    return x