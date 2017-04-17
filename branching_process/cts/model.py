# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Poisson point process penalised likelihood regression.
"""

try:
    import autograd
    import autograd.numpy as np
    import autograd.scipy as sp
    have_autograd = True
except ImportError as e:
    import numpy as np
    import scipy as sp
    have_autograd = False

from scipy.stats import gaussian_kde
from . import influence
from . import background


def _as_mu_args(mu=None, omega=None, tau=None, **kwargs):
    """
    utility function to convert model arguments to kernel arguments.
    This renames omga and mu, and *ignores* tau.
    """
    kwargs = dict(mu=mu, **kwargs)
    if omega is not None:
        kwargs['kappa'] = omega
    if mu is not None:
        kwargs['mu'] = mu
    return kwargs


def _as_phi_args(kappa=None, tau=None, **kwargs):
    """
    utility function to convert model arguments to kernel arguments
    """
    kwargs = dict(kappa=kappa, **kwargs)
    if tau is not None:
        kwargs['tau'] = tau
    return kwargs


def lam(
        ts,
        eval_ts=None,
        bw=1.0):
    """
    """
    if eval_ts is None:
        eval_ts = ts
    fn = gaussian_kde(
        ts,
        bw * (np.amax(ts)-np.amin(ts))/ts.size
        # * (ts.size**(-0.8))
    )
    return fn(eval_ts) * ts.size


def lam_hawkes(
        ts,
        mu=0.0,
        eval_ts=None,
        max_floats=1e8,
        phi_kernel=None,
        mu_kernel=None,
        **kwargs):
    """
    Intensity of Hawkes process given time series and parameters.
    Memory-hungry per default; could be improved with numba.
    """
    ts = np.asfarray(ts).ravel()

    if eval_ts is None:
        eval_ts = ts
    eval_ts = np.asfarray(eval_ts).ravel()
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = background.as_background_kernel(mu_kernel)
    if ((ts.size) * (eval_ts.size)) > max_floats:
        return _lam_hawkes_lite(
            ts=ts,
            phi_kernel=phi_kernel,
            mu_kernel=mu_kernel,
            eval_ts=eval_ts,
            **kwargs
        )
    mu_kwargs = _as_mu_args(mu=mu, **kwargs)
    phi_kwargs = _as_phi_args(**kwargs)
    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    endo = phi_kernel(
        deltas.ravel(),
        **phi_kwargs
    ).reshape(deltas.shape) * mask
    exo = mu_kernel(
        eval_ts, **mu_kwargs
    )
    return endo.sum(0) + exo


def _lam_hawkes_lite(
        ts,
        eval_ts,
        mu_kernel,
        phi_kernel,
        t_start=0.0,
        **kwargs):
    """
    Intensity of Hawkes process given time series and parameters.
    Memory-lite version. CPU-hungry, could be improved with numba.

    Uses assignment so may need to be altered for differentiability.
    """
    endo = np.zeros_like(eval_ts)
    mu_kwargs = _as_mu_args(mu=mu, **kwargs)
    phi_kwargs = _as_phi_args(**kwargs)
    for i in range(eval_ts.size):
        deltas = eval_ts[i] - ts
        mask = deltas > 0.0
        endo[i] = np.sum(phi_kernel(deltas, **phi_kwargs) * mask)
    exo = mu_kernel(eval_ts, **mu_kwargs)
    return endo + exo


def big_lam_hawkes(
        ts,
        eval_ts,
        mu=1.0,
        t_start=0.0,
        mu_kernel=None,
        phi_kernel=None,
        **kwargs
        ):
    """
    True integrated intensity of hawkes process.
    since you are probably evaluating this only at one point,
    this is only available in a vectorised high-memory version.
    """
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = background.as_background_kernel(mu_kernel)
    ts = np.asfarray(ts).ravel()
    mu_kwargs = _as_mu_args(mu=mu, **kwargs)
    phi_kwargs = _as_phi_args(**kwargs)
    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    big_endo = phi_kernel.integrate(
        deltas.ravel(),
        **phi_kwargs
    ).reshape(deltas.shape) * mask
    big_exo = (
        mu_kernel.integrate(eval_ts, **mu_kwargs) -
        mu_kernel.integrate(t_start, **mu_kwargs)
    )
    return big_endo.sum(0) + big_exo


def loglik(
        ts,
        phi=None,
        mu=1.0,
        t_start=0.0,
        t_end=None,
        eval_ts=None,
        phi_kernel=None,
        mu_kernel=None,
        **kwargs):
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = background.as_background_kernel(mu_kernel)

    if t_end is None:
        t_end = ts[-1]
    # as an optimisation we allow passing in an eval_ts array,
    # in which case t_start and t_end are ignored.
    if eval_ts is None:
        if t_end > ts[-1]:
            eval_ts = np.concatenate((ts[ts > t_start], [t_end]))
        else:
            eval_ts = ts[np.logical_and((ts > t_start), (ts < t_end))]

    lam = lam_hawkes(
        ts=ts,
        mu=mu,
        phi_kernel=phi_kernel,
        mu_kernel=mu_kernel,
        eval_ts=eval_ts,
        **kwargs
    )
    big_lam = big_lam_hawkes(
        ts=ts,
        mu=mu,
        phi_kernel=phi_kernel,
        mu_kernel=mu_kernel,
        t_start=t_start,
        eval_ts=np.array(t_end),
        **kwargs
    )
    # from IPython.core.debugger import Tracer; Tracer()()

    return np.sum(np.log(lam)) - big_lam
