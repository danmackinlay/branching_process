"""
Where we transform stateless long-memory time series models into
predictor/response matrices
"""

import numpy as np
from scipy.signal import convolve


def causal_pad(basis):
    """
    scipy's convolve dosn't enforce causality;
    (i.e. convolving only the history of a signal)
    We need to do that by adding zeroes to the convolution matrix
    """
    causal_basis = np.zeros((basis.shape[0], basis.shape[1]+1))
    causal_basis[:, 1:] = basis
    return causal_basis


def history_expand(signal, basis, pad=True):
    if pad:
        basis = causal_pad(basis)
    # convolve does not broadcast, so we do it manually
    n_bases, _ = basis.shape
    n_steps = signal.size
    out = np.zeros((n_bases, n_steps-1))
    for i in range(n_bases):
        out[i, :] = convolve(
            signal, basis[i, :], mode='full'
        )[1:n_steps]
    return out
