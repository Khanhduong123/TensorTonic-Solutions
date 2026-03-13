import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[: , np.newaxis]
    i = np.arange(d_model // 2)[np.newaxis, :]
    denom = np.power(base , (2 * i) / d_model)
    pe = np.zeros((seq_len, d_model))

    pe[: , 0:2*(d_model//2):2] = np.sin(pos / denom)
    pe[: , 1:2*(d_model//2):2] = np.cos(pos/denom)

    if d_model % 2 == 1:
        # Calculate the specific denominator for the very last index
        last_denom = np.power(base, (d_model - 1) / d_model)
        pe[:, -1] = np.sin(pos / last_denom).squeeze()
    return pe