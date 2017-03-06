"""
INAR/branching process simulation in discrete time.
"""
import numpy as np
import warnings
from scipy.stats import poisson, geom
from .dist import gpd


def sim_inar(
        immigrants=np.array([]),
        mu=0.0,
        phi=None,
        kappa=0.5,
        conditional_rvs=None,
        end=None,
        return_rates=False,
        return_partials=False,
        dispersion=0.0,
        eta=1.0):
    """
    Simulate self-exciting INAR process

    :type phi: numpy.array
    :param phi: influence kernel for counts,
      a vector of weights mapping count history vectors to a scalar.

    :type kappa: function
    :param kappa: scale factor for the phi kernel ratio.

    :type conditional: function
    :param conditional: function mapping from rate param to rv dist

    :type immigrants: np.array
    :param immigrants: An array of seed counts.

    :type mu: float
    :param mu: mean rate

    :type end: int
    :param end: stop simulation here

    :return: vector of simulated child event counts
    :rtype: numpy.array
    """

    if end is None:
        end = immigrants.size
    if phi is None:
        phi = geom(0.5).pmf(np.linspace(50))

    phi *= eta

    if conditional_rvs is None or conditional_rvs == 'poisson':
        if dispersion != 0.0:
            warnings.warn(
                'Warning! dispersion parameter supplied to Poisson '
            )
        conditional_rvs = (lambda rate: poisson.rvs(
                mu=rate
            )
        )
    elif conditional_rvs == 'gpd':
        conditional_rvs = (lambda rate: gpd.rvs(
                mu=rate, eta=dispersion
            )
        )
    else:
        if dispersion != 0.0:
            warnings.warn(
                'Warning! dispersion parameter supplied, but'
                'we have our own distribution'
            )
    rev_phi = phi[::-1]
    total_counts = np.zeros(end, dtype='int')
    total_counts[:immigrants.size] = immigrants
    rates = np.zeros(total_counts.size)
    partials = []
    # step 0 has 0-length arrays; we special case it
    next_increment = conditional_rvs(mu)
    total_counts[0] += next_increment

    for i in range(1, end):
        window_len = min(i, rev_phi.size)
        # print(window_len, "hist", i-window_len,i, 'k', -window_len)
        # convolve kernel with history so far
        partial = (
            total_counts[
                i-window_len:i
            ] * rev_phi[
                -window_len:
            ]
        )
        ar_p = mu + kappa * (partial.sum())
        rates[i] = ar_p
        if return_partials:
            partial = np.pad(partial, (rev_phi.size-window_len, 0), 'constant')
            partials.append(partial.reshape(-1, 1))

        next_increment = conditional_rvs(ar_p)
        total_counts[i] += next_increment
    if return_partials:
        return total_counts, rates, np.concatenate(partials, 1)
    if return_rates:
        return total_counts, rates
    return total_counts
