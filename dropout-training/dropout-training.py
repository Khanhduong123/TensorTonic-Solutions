import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    The pattern array contains 0 for dropped and 1/(1-p) for kept elements.
    """
    x = np.asarray(x)
    
    # Handle the edge case where p=1.0 (all dropped) to avoid division by zero
    if p >= 1.0:
        pattern = np.zeros_like(x, dtype=float)
        return x * pattern, pattern

    if rng is None:
        rng = np.random.default_rng()

    # 1. Define the survival probability (keep_prob)
    keep_prob = 1.0 - p
    
    # 2. Generate random values and create a binary mask (0 or 1)
    # Most references use '< keep_prob'
    binary_mask = (rng.random(x.shape) < keep_prob).astype(float)
    
    # 3. Create the pattern array as per Hint 2:
    # Elements are either 0 or 1/(1-p)
    dropout_pattern = binary_mask / keep_prob
    
    # 4. Multiply input by this pre-scaled pattern
    output = x * dropout_pattern
    
    return output, dropout_pattern