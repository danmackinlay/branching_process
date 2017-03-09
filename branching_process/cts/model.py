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

from . import influence


def lam_hawkes(
        ts,
        mu,
        phi_kernel=0.0,
        mu_kernel=0.0,
        eta=1.0,
        eval_ts=None,
        max_floats=1e8,
        phi_kwargs={},
        mu_kwargs={},
        **kwargs):
    """
    True intensity of Hawkes process.
    Memory-hungry per default; could be improve with numba.
    """
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = influence.as_influence_kernel(mu_kernel)
    ts = np.asfarray(ts).ravel()
    mu_kwargs = dict(
        mu=mu,
        **mu_kwargs
    )

    if eval_ts is None:
        eval_ts = ts
    eval_ts = np.asfarray(eval_ts).ravel()
    if ((ts.size) * (eval_ts.size)) > max_floats:
        return _lam_hawkes_lite(
            ts=ts,
            phi_kernel=phi_kernel,
            mu_kernel=mu_kernel,
            eta=eta,
            eval_ts=eval_ts,
            phi_kwargs=phi_kwargs,
            mu_kwargs=mu_kwargs
        )
    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    endo = phi_kernel(
        deltas.ravel(),
        **phi_kwargs
    ).reshape(deltas.shape) * mask
    exo = mu_kernel(
        eval_ts, **mu_kwargs
    )
    return endo.sum(0) * eta + exo


def _lam_hawkes_lite(
        ts,
        eval_ts,
        mu_kernel,
        phi_kernel,
        eta=1.0,
        start=0.0,
        phi_kwargs={},
        mu_kwargs={},
        **kwargs):
    """
    True intensity of Hawkes process.
    Memory-lite version. CPU-hungry, could be improved with numba.

    Uses assignment so may need to be altered for differentiability.
    """
    endo = np.zeros_like(eval_ts)
    deltas = np.zeros_like(ts)
    mask = np.zeros_like(ts)
    for i in range(eval_ts.size):
        deltas[:] = eval_ts[i] - ts
        mask[:] = deltas > 0.0
        endo[i] = np.sum(phi_kernel(deltas, **phi_kwargs) * mask)
    exo = mu_kernel(eval_ts, **mu_kwargs)
    return endo * eta + exo


def big_lam_hawkes(
        ts,
        eval_ts,
        mu,
        phi_kernel,
        mu_kernel=1.0,
        eta=1.0,
        start=0.0,
        phi_kwargs={},
        mu_kwargs={},
        **kwargs
        ):
    """
    True integrated intensity of hawkes process.
    since you are probably evaluating this only at one point,
    this is only available in a vectorised high-memory version.
    """
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = influence.as_influence_kernel(mu_kernel)
    ts = np.asfarray(ts).ravel()
    mu_kwargs = dict(
        mu=mu,
        **mu_kwargs
    )
    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    big_endo = phi_kernel.integrate(
        deltas.ravel(),
        **phi_kwargs
    ).reshape(deltas.shape) * mask
    big_exo = (
        mu_kernel.integrate(eval_ts, **mu_kwargs) -
        mu_kernel.integrate(start, **mu_kwargs)
    )
    return big_endo.sum(0) * eta + big_exo


def loglik(
        ts,
        phi_kernel=None,
        mu_kernel=1.0,
        mu=1.0,
        eta=1.0,
        start=0.0,
        end=None,
        eval_ts=None,
        omega=[],
        mu_kwargs={},
        phi_kwargs={}):
    phi_kernel = influence.as_influence_kernel(phi_kernel)
    mu_kernel = influence.as_influence_kernel(mu_kernel)

    if phi_kernel is None:
        phi_kernel = phi_kernel
    if end is None:
        end = ts[-1]
    # as an optimisation we allow passing in an eval_ts array,
    # in which case start and end are ignored.
    if eval_ts is None:
        if end > ts[-1]:
            eval_ts = np.concatenate((ts[ts > start], [end]))
        else:
            eval_ts = ts[np.logical_and((ts > start), (ts < end))]

    lam = lam_hawkes(
        ts=ts,
        mu=mu,
        phi_kernel=phi_kernel,
        mu_kernel=mu_kernel,
        eta=eta,
        eval_ts=eval_ts,
        phi_kwargs=phi_kwargs,
        mu_kwargs=mu_kwargs
    )
    big_lam = big_lam_hawkes(
        ts=ts,
        mu=mu,
        phi_kernel=phi_kernel,
        mu_kernel=mu_kernel,
        start=start,
        eta=eta,
        eval_ts=np.array(end),
        phi_kwargs=phi_kwargs,
        mu_kwargs=mu_kwargs
    )

    return np.sum(np.log(lam)) - big_lam
