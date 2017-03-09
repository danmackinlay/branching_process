
"""
Influence kernels in the discrete case are just matrices;
We return them with timesteps as second index, one basis kernel in each row.
"""
import numpy as np
from scipy.special import gammaln
import math


def ar_basis(max_m=20, *args, **kwargs):
    """
    Classic AR(m) basis, a.k.a. histogram basis
    """
    return np.eye(max_m)


def decaying_basis(
        max_m=10,
        clip_t=None,
        ratio=2,
        *args, **kwargs):
    """
    nonparametric basis with decaying terms
    """
    reps = ratio ** np.arange(max_m).astype("int")
    kernel = np.repeat(np.eye(max_m), reps, axis=1)
    return kernel / (kernel.sum(1).reshape(-1, 1))


def poisson_basis(
        max_m=20,
        t_start=1,
        clip_t=None,
        spacing=2.0,
        epsilon=1e-10,
        *args, **kwargs):
    taus = []
    prev_tau = t_start
    for i in range(max_m + 1):
        sd = math.sqrt(prev_tau)
        prev_tau = sd * spacing + prev_tau
        taus.append(prev_tau)
    max_support = taus.pop() * 1.5
    max_t = min(clip_t or max_support, max_support)
    taus = np.array(taus).reshape(-1, 1)
    ts = np.arange(max_t)
    log_kernel = -taus + ts*np.log(taus) - gammaln(ts + 1)
    kernel = np.exp(log_kernel)
    # null denormals
    kernel[kernel < epsilon] = 0.0
    # normalise away null and truncation problems
    return kernel / (kernel.sum(1).reshape(-1, 1))


def geom_basis(max_m=10, t_start=1, clip_t=None, spacing=1.0, *args, **kwargs):
    taus = []
    prev_tau = t_start
    for i in range(max_m+1):
        taus.append(prev_tau)
        sd = prev_tau
        prev_tau += sd * spacing
    max_support = taus.pop() * 1.5
    max_t = min(clip_t or max_support, max_support)

    taus = np.array(taus).reshape(-1, 1)
    if max_t is None:
        max_t = math.floor(taus[-1, 0]*1.5)
    ps = 1.0/taus
    ips = 1.0 - ps
    ts = np.arange(max_t)
    # Construct it in log-domain or it explodes after a few coefficients.
    log_kernel = ts * np.log(ips) + ps
    par_ker = np.exp(log_kernel)
    # Normalise manually, since we may in any case be truncating
    return par_ker / (par_ker.sum(1).reshape(-1, 1))
