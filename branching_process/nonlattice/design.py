"""
Time series interpolation and smoothing
"""
import numpy as np
from scipy.interpolate import PPoly, PchipInterpolator, UnivariateSpline
from . import convert
from . import weighted_linear_model
from importlib import reload
weighted_linear_model = reload(weighted_linear_model)
convert = reload(convert)


def predict_intensity(big_n, obs_t, mu, basis_lag, coef):
    """
    This should return predicted intensity at observation times
    between successive observations
    """
    pred, resp, _ = (
        design_stepwise(
            obs_t=obs_t,
            basis_lag=basis_lag,
            big_n_hat=big_n,
            cum_obs=big_n(obs_t)
        )
    )
    return mu + np.sum(coef * pred, 1)


def predict_increment(big_n, obs_t, mu, basis_lag, coef):
    """
    This should return predicted increments
    between successive observations
    """
    rate_hat = predict_intensity(
        big_n, obs_t, mu, basis_lag, coef
    )
    increment_size = np.diff(obs_t)
    return rate_hat * increment_size


class DiracInterpolant:
    """
    interpolation for approximate spike train using weighted deltas
    """
    def __init__(
            self,
            spike_t,
            cum_obs,
            obs_t,
            spike_bin,
            spike_counts,
            spike_weight,
            spike_lattice=None,
            step_size=None,
            lambda_hat=None,
            spike_cum_weight=None):
        self.spike_t = spike_t
        self.cum_obs = cum_obs
        self.obs_t = obs_t
        self.spike_bin = spike_bin
        self.spike_counts = spike_counts
        self.spike_weight = spike_weight
        self.delta_obs = np.diff(cum_obs)
        if spike_lattice is None:
            spike_lattice = np.zeros(spike_weight.size + 1)
            spike_lattice[0] = obs_t[0]
            spike_lattice[-1] = obs_t[-1]
            spike_lattice[1:-1] = (spike_t[0:-1] + spike_t[1:])/2.0
        self.spike_lattice = spike_lattice
        if step_size is None:
            step_size = np.diff(spike_lattice)
        self.step_size = step_size
        if spike_cum_weight is None:
            spike_cum_weight = np.zeros(spike_weight.size + 1)
            spike_cum_weight[:-1] = np.cumsum(spike_weight) + cum_obs[0]
            spike_cum_weight[-1] = cum_obs[-1]
        self.spike_cum_weight = spike_cum_weight
        if lambda_hat is None:
            lambda_hat = spike_weight/step_size
        self.lambda_hat = lambda_hat

    def __call__(
            self,
            quad_times):
        """
        integrate
        """
        last_spike_before = np.minimum(
            np.searchsorted(
                self.spike_t, quad_times
            ),
            self.spike_t.size - 1
        )
        return self.spike_cum_weight[last_spike_before]

    def set_intensity_(self, i, intensity):
        """
        update intensity in place via the derivative
        Don't touch the obs_time stuff for performance reasons.
        """
        weight = intensity * self.step_size[i]
        self.set_weight_(i, weight)

    def set_weight_(self, i, weight):
        """
        update cumulative count in place
        """
        self.spike_weight[i] = weight
        np.cumsum(
            self.spike_weight,
            out=self.spike_cum_weight)
        np.add(
            self.spike_weight,
            self.cum_obs[0],
            out=self.spike_weight)
        self.lambda_hat = self.spike_weight/self.step_size


def dirac_interpolant_from_obs(
        obs_t,
        cum_obs,
        step_size=None,
        strategy='spike'):
    """
    return arrays to construct dirac interpolant.
    Essentially, this is numerical quadrature approximation.

    Could random spike locations; would that be less biassed?
    Or low discrepancy sequences?
    """
    obs_t = np.asfarray(obs_t)
    cum_obs = np.asfarray(cum_obs)

    if step_size is None:
        step_size = (obs_t[-1] - obs_t[0]) / (obs_t.size * 8.0)

    # generate steps
    spike_t = np.arange(
        obs_t[0] - step_size/2,
        obs_t[-1],
        step_size, dtype='float'
    )

    # Identify observations that are too close to be seen by the grid
    # and throw them out (better approach?)
    (obs_t_i) = np.unique(
        np.searchsorted(obs_t, spike_t)
    )

    spike_t = spike_t[1:]
    # bin_edge_i = np.searchsorted(spike_t, obs_t[1:-1])
    thinned_obs_t = obs_t[obs_t_i]
    thinned_cum_obs = cum_obs[obs_t_i]

    (thinned_obs_t_i, spike_t_i, spike_bin, spike_counts) = np.unique(
        np.searchsorted(thinned_obs_t, spike_t),
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    if strategy == 'uniform':
        # first-pass weight approximation - piecewise constant
        # spike_weight = (delta_obs/spike_counts)[spike_bin]
        spike_weight = np.ones_like(spike_t)
    elif strategy == 'spike':
        spike_weight = np.zeros_like(spike_t)
        spike_weight[spike_t_i] = 1
    elif strategy == 'random':
        # heavy-tailed random noise
        spike_weight = 1.0/(np.random.rand(spike_t.size) + 1e-8)**2
    else:
        raise Exception('unknown strategy "{}"'.format(strategy))\

    big_n_hat = DiracInterpolant(
        spike_t=spike_t,
        cum_obs=thinned_cum_obs,
        obs_t=thinned_obs_t,
        spike_bin=spike_bin,
        spike_counts=spike_counts,
        spike_weight=spike_weight
    )
    return reweight_dirac_interpolant(big_n_hat)


def reweight_dirac_interpolant(
        big_n_hat,
        n_hat_arr=None):
    """
    redistribute sample weights to match observed increments
    """

    if n_hat_arr is None:
        n_hat_arr = big_n_hat.spike_weight
    # count the n increments
    obs_increments = np.zeros_like(big_n_hat.delta_obs)
    np.add.at(obs_increments, big_n_hat.spike_bin, n_hat_arr)

    correction = (
        big_n_hat.delta_obs / obs_increments
    )[big_n_hat.spike_bin]
    new_spike_weight = n_hat_arr * correction
    return DiracInterpolant(
        spike_t=big_n_hat.spike_t,
        cum_obs=big_n_hat.cum_obs,
        obs_t=big_n_hat.obs_t,
        spike_bin=big_n_hat.spike_bin,
        spike_counts=big_n_hat.spike_counts,
        spike_weight=new_spike_weight,
        spike_lattice=big_n_hat.spike_lattice,
    )


def copy_dirac_interpolant(
        big_n_hat,
        n_hat_arr=None):
    """
    copy interpolant for in-place modification
    """
    return DiracInterpolant(
        spike_t=big_n_hat.spike_t.copy(),
        cum_obs=big_n_hat.cum_obs.copy(),
        obs_t=big_n_hat.obs_t.copy(),
        spike_bin=big_n_hat.spike_bin.copy(),
        spike_counts=big_n_hat.spike_counts.copy(),
        spike_weight=big_n_hat.spike_weight.copy(),
        spike_lattice=big_n_hat.spike_lattice.copy(),
    )


def linear_interp(obs_t, cum_obs):
    """
    Construct a linear count interpolant
    (which for monotonic counts is a stepwise rate)
    """
    obs_t = np.asarray(obs_t)
    cum_obs = np.asarray(cum_obs)
    coeffs = np.zeros((2, cum_obs.size + 1))
    knots = np.zeros(obs_t.size + 2)
    # interpolating integral construction is awkward
    # because the spline constructors only like cubics,
    # so I build the PPoly manually

    knots[1:-1] = obs_t

    # extend with null counts
    # so that it extrapolates, but conservatively
    knots[0] = obs_t[0] - 1
    knots[-1] = obs_t[-1] + 1

    # rate
    coeffs[0, 1:-1] = (
        cum_obs[1:] - cum_obs[:-1]
    ) / (
        obs_t[1:] - obs_t[:-1]
    )

    # step
    coeffs[1, 1:] = cum_obs
    # edge
    coeffs[1, 0] = cum_obs[0]

    big_n_hat = PPoly(
        coeffs,
        knots,
        extrapolate=True)
    return big_n_hat


def cubic_interp(obs_t, cum_obs):
    """
    Construct a cubic count interpolant
    (which for monotonic counts is a quadratic rate)
    """

    # extend with null counts
    # so that it extrapolates, but conservatively
    obs_t = np.concatenate([
        [obs_t[0] - 2, obs_t[0] - 1],
        obs_t,
        [obs_t[-1] + 1, obs_t[-1] + 2]
    ])
    cum_obs = np.concatenate([
        [cum_obs[0], cum_obs[0]],
        cum_obs,
        [cum_obs[-1], cum_obs[-1]]
    ])

    big_n_hat = PPoly.from_bernstein_basis(
        PchipInterpolator(
            obs_t,
            cum_obs,
            extrapolate=True
        )
    )
    return big_n_hat


def cubic_smooth(obs_t, cum_obs, smoothing=1.0):
    """
    Construct a cubic count smoother
    (which for monotonic counts is a quadratic rate)

    This is a hack, and doesn't guarantee monotonicity;
    Be careful.

    the default smoothing parameter of scipy is abysmal.
    We make a better default.
    """
    obs_t = np.asfarray(obs_t)
    cum_obs = np.asfarray(cum_obs)
    # Here's a bad smoothing factor
    # which ignores the fact there is a trend by construction
    s0 = np.var(cum_obs)
    s1 = UnivariateSpline(
        obs_t,
        cum_obs,
        s=s0 * smoothing
    ).get_residual()/(obs_t.size)
    big_n_hat = UnivariateSpline(
        obs_t,
        np.asfarray(cum_obs),
        s=s1*smoothing,
        ext=3
    )
    return big_n_hat


def interpolate(
        obs_t,
        cum_obs,
        cum_interp='linear',
        smoothing=1.0,
        step_size=None,
        strategy='random',
        ):
    """
    generic nonparametric interpolation of curve.
    """
    if cum_interp == 'linear':
        big_n_hat = linear_interp(obs_t, cum_obs)
    elif cum_interp == 'cubic':
        big_n_hat = cubic_interp(obs_t, cum_obs)
    elif cum_interp == 'cubic_smooth':
        big_n_hat = cubic_smooth(
            obs_t, cum_obs,
            smoothing=smoothing)
    elif cum_interp == 'dirac':
        big_n_hat = dirac_interpolant_from_obs(
            obs_t, cum_obs,
            step_size=step_size,
            strategy=strategy)
    else:
        raise ValueError('unknown interpolant "{}"'.format(cum_interp))

    return big_n_hat


def design_stepwise(
        obs_t,
        cum_obs,
        basis_lag,
        big_n_hat,
        sample_weight='bermanturner',
        ):
    """
    Convolve an interpolator function
    wrt a piecewise-constant influence kernel
    in order to get predictors and responses.

    The interpolator is assumed to be the integral of the observations
    to be convolved.

    TODO: separate predictors/responses
    """
    basis_lag = np.asfarray(basis_lag)
    obs_t = np.asarray(obs_t)
    cum_obs = np.asarray(cum_obs)
    delta_t = np.diff(obs_t)
    responses = np.diff(cum_obs)/delta_t

    quad_times = obs_t[:-1].reshape(-1, 1) - basis_lag.reshape(1, -1)
    cum_endo = big_n_hat(quad_times)
    # import pdb; pdb.set_trace()
    predictors = -np.diff(cum_endo, 1)
    if sample_weight == 'bermanturner':
        # weight according to the Berman-Turner device, where longer intervals
        # are more important
        sample_weight = delta_t
    elif sample_weight is None or sample_weight == 'equal':
        # But maybe you want to weight naively
        # because longer interval should have higher variance
        sample_weight = np.ones_like(delta_t)
    # otherwise you can pass in whatever weights you like

    return predictors, responses, sample_weight
