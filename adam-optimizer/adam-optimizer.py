import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.asarray(m)
    v= np.asarray(v)
    # t = np.asarray(t)
    grad = np.asarray(grad)
    param = np.asarray(param)
    
    #B1: Update first and second momentum
    mt = beta1 * m + (1- beta1) * grad
    vt = beta2 * v + (1 - beta2) * grad**2

    #B2: Calculate bias correction
    m_hat = mt / (1 - beta1**t)
    v_hat = vt / (1 - beta2**t)

    #B3: parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param_new, mt, vt
