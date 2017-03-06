"""
Utilities for evaluating estimators, or comparing two different estimators
through simulation.
"""
from . import sim_nonlattice
from . import sim_cts
import numpy as np
from scipy.stats import expon
from importlib import reload
sim_nonlattice = reload(sim_nonlattice)
sim_cts = reload(sim_cts)


def fits_compare(
        model_1,
        model_2=None,
        rate_1=1.0,
        rate_2=None,
        obs_times_1='poisson',
        obs_times_2=None,
        sim=sim_cts.sim_hawkes,
        phi=lambda t: expon(scale=5.0).pdf(t),
        mu=9.0,
        eta=0.9,
        start=0.0,
        end=300,
        n_iter=1000,
        basis_lag=1.0,
        penalty_weight='adaptive',
        sample_weight='bermanturner',
        **extra_fit_args
        ):

    obs_times_2 = obs_times_2 if obs_times_2 else obs_times_1
    rate_2 = rate_2 if rate_2 else rate_1

    fits_1 = []
    fits_2 = []

    for i in range(n_iter):
        timestamps = np.sort(sim(
            phi=phi, mu=mu, eta=eta, start=start, end=end
        ))
        fits_1.append(
            fit_and_analyse(
                model_1,
                timestamps,
                obs_times_1,
                rate_1,
                basis_lag=basis_lag,
                penalty_weight=penalty_weight,
                sample_weight=sample_weight,
                **extra_fit_args
            )
        )

        if model_2 is not None:
            fits_2.append(
                fit_and_analyse(
                    model_2,
                    timestamps,
                    obs_times_2,
                    rate_2,
                    basis_lag=basis_lag,
                    penalty_weight=penalty_weight,
                    sample_weight=sample_weight,
                    **extra_fit_args
                )
            )

    res = dict(
        fits_1=flatten_fits(fits_1),
        fits_2=flatten_fits(fits_2),
        basis_lag=model_1.basis_lag_,  # presumed all the same
    )
    return res


def fit_and_analyse(
        model,
        timestamps,
        obs_t,
        rate,
        **kwargs):

    obs_t, cum_obs = sim_nonlattice.quantize_timestamps(
        timestamps,
        obs_t=obs_t,
        rate=rate)
    model.fit(
        obs_t,
        cum_obs,
        **kwargs)
    inc_prediction = model.predict(
        obs_t,
    )
    residual = np.diff(cum_obs) - inc_prediction
    # oracle calc also?
    # i.e. estimated versus actual compensator?
    return dict(
        intercept=model.intercept_,
        coef=model.coef_.ravel(),
        basis_lag=model.basis_lag_.ravel(),
        residual=residual
    )


def flatten_fits(fit_list):
    """
    flatten
    """
    if len(fit_list) == 0:
        return
    n_row = len(fit_list)
    n_basis_col = max(
        np.amax([
            len(f['coef']) for f in fit_list
        ]),
        np.amax([
            len(f['basis_lag']) for f in fit_list
        ]),
    )
    coef = np.zeros((n_row, n_basis_col))
    for i, c in enumerate(fit_list):
        coef[i, :c['coef'].size] = c['coef'].ravel()
    # Is this right? or should I take mean? Or adjust for Poisson variance?
    sq_residual = np.array([
        np.sum(np.square(r['residual']))
        for r in fit_list
    ])
    intercept = np.array([
        s['intercept']
        for s in fit_list
    ])
    return dict(
        coef=coef,
        intercept=intercept,
        sq_residual=sq_residual,
    )
