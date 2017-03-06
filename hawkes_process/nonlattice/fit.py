"""
Poisson time series penalised likelihood regression
via the Berman Turner device
"""

from . import weighted_linear_model
from . import design_nonlattice as design
from math import ceil
import numpy as np

from importlib import reload
design = reload(design)


class NonLatticeOneShot:
    """
    the simplest device.
    Uses a stepwise-constant quadrature rule and non-adaptive
    smoothing.
    """
    def __init__(
            self,
            positive=True,
            normalize=False,
            wlm=None,
            wlm_factory='WeightedLassoLarsCV',
            cum_interp='linear',
            smoothing=1.0,  # only for spline smoother
            step_size=0.25,  # only for dirac interpolant
            strategy='random',  # only for dirac interpolant
            *args, **kwargs):
        if wlm is None:
            # Allow reference by class for easy serialization
            if isinstance(wlm_factory, str):
                wlm_factory = getattr(weighted_linear_model, wlm_factory)
            self.wlm = wlm_factory(
                positive=positive,
                normalize=normalize,
                *args, **kwargs
            )
        else:
            self.wlm = wlm
        self.big_n_hat_ = None
        self.cum_interp = cum_interp
        self.smoothing = smoothing
        self.strategy = strategy
        self.step_size = step_size

    def fit(
            self,
            obs_t,
            cum_obs,
            basis_lag=1.0,
            penalty_weight='adaptive',
            sample_weight='bermanturner',
            max_basis_span=float('inf'),
            big_n_hat=None,
            *args, **kwargs):

        self.obs_t_ = obs_t
        self.cum_obs_ = cum_obs

        if np.isscalar(basis_lag):
            # scalars are a bin width
            basis_span = min(
                (np.amax(obs_t) - np.amin(obs_t))/2.0,
                max_basis_span
            )
            n_bins = ceil(basis_span/basis_lag)
            self.basis_lag_ = np.arange(n_bins+1) * basis_lag
        else:
            self.basis_lag_ = basis_lag
        if big_n_hat is None:
            self.big_n_hat_ = self.predict_big_n()

        (
            self.inc_predictors_,
            self.inc_response_,
            self.inc_sample_weight_
        ) = (
            design.design_stepwise(
                obs_t=self.obs_t_,
                cum_obs=self.cum_obs_,
                basis_lag=self.basis_lag_,
                big_n_hat=self.big_n_hat_,
                sample_weight=sample_weight
            )
        )

        self.wlm.fit(
            X=self.inc_predictors_,
            y=self.inc_response_,
            sample_weight=self.inc_sample_weight_,
            penalty_weight=penalty_weight,
            *args, **kwargs
        )

    def predict_intensity(self, obs_t=None):
        """
        This should return forward-predicted intensity
        based on the fitted histogram, up to the last observations
        before the given times.
        """
        return design.predict_increment(
            big_n=self.big_n_hat_,
            obs_t=obs_t if obs_t is not None else self.obs_t_,
            mu=self.intercept_,
            basis_lag=self.basis_lag_,
            coef=self.coef_)

    def predict(self, obs_t=None):
        """
        This should return predicted increments
        based on the fitted histogram, up to the last observations
        before the given times.
        """

        return design.predict_increment(
            big_n=self.big_n_hat_,
            obs_t=obs_t if obs_t is not None else self.obs_t_,
            mu=self.intercept_,
            basis_lag=self.basis_lag_,
            coef=self.coef_)

    def predict_big_n(self, obs_t=None):
        """
        This should return predicted increment interpolant
        based on the fitted histogram, up to the last observations
        before the given times.
        """
        return design.interpolate(
            obs_t=self.obs_t_,
            cum_obs=self.cum_obs_,
            cum_interp=self.cum_interp,
            smoothing=self.smoothing,
            step_size=self.step_size,
            strategy=self.strategy,
        )

    @property
    def coef_(self):
        return self.wlm.coef_

    @property
    def eta_(self):
        return np.sum(self.coef_)

    @property
    def intercept_(self):
        return self.wlm.intercept_

    @property
    def alpha_(self):
        return self.wlm.alpha_

    @property
    def n_iter_(self):
        return self.wlm.n_iter_


class NonLatticeIterative(NonLatticeOneShot):
    """
    repeatedly forward-smooth to find optimal interpolant.
    TODO: This doesn't do backwards losses
    """
    def __init__(
            self,
            *args, **kwargs):

        super().__init__(
            cum_interp='dirac',
            strategy='random',
            *args, **kwargs)

    def fit(
            self,
            obs_t,
            cum_obs,
            basis_lag=1.0,
            penalty_weight='adaptive',
            sample_weight='bermanturner',
            max_basis_span=float('inf'),
            big_n_hat=None,
            max_iter=3,
            *args, **kwargs):
        inner_model = NonLatticeOneShot(
            wlm=self.wlm,
            cum_interp='linear',
        )
        self.inner_model = inner_model
        self.obs_t_ = obs_t
        self.cum_obs_ = cum_obs

        if np.isscalar(basis_lag):
            # scalars are a bin width
            basis_span = min(
                (np.amax(obs_t) - np.amin(obs_t))/2.0,
                max_basis_span
            )
            n_bins = ceil(basis_span/basis_lag)
            self.basis_lag_ = np.arange(n_bins+1) * basis_lag
        else:
            self.basis_lag_ = basis_lag
        if big_n_hat is None:
            self.big_n_hat_ = self.predict_big_n()

        for i in range(max_iter):
            print('i', i, max_iter)
            inner_model.fit(
                obs_t=self.big_n_hat_.spike_lattice,
                cum_obs=self.big_n_hat_.spike_cum_weight,
                *args,
                **kwargs)

            n_hat_arr = inner_model.predict(
                obs_t=self.big_n_hat_.spike_lattice,
            )
            self.big_n_hat_ = design.reweight_dirac_interpolant(
                self.big_n_hat_,
                n_hat_arr
            )
