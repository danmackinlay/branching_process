import numpy as np
from numpy import random
from warnings import warn


def sim_poisson(
        mu,
        start=0.0,
        end=1.0):
    """
    Simulate constant-rate Poisson process.

    :type mu: float
    :param mu: rate of the process.

    :type start: float
    :param start: start simulating immigrants at this time

    :type end: float
    :param end: no events after this time

    :return: vector of simulated event times on [start, end], unsorted.
    :rtype: numpy.array
    """
    timespan = end-start
    N = random.poisson(lam=mu*timespan)
    return start + random.rand(N)*timespan


def sim_piecewise_poisson(
        mu_v,
        t_v):
    """
    Simulate piecewise constant-rate Poisson process.

    :type mu_v: np.array
    :param mu: background rate of the immigrant process,
      assumed to be stepwise defined over intervals.
      Any of these after the last ``t_v.size-1``
      are ignored.

    :type t_v: np.array
    :param t_v: times of jumps in the step process.

    :return: vector of simulated event times on [t_v[0], t_v[-1]], unsorted.
    :rtype: numpy.array
    """
    mu_v = np.asarray(mu_v)
    t_v = np.asarray(t_v)
    n_steps = t_v.size - 1

    return np.concatenate([
        sim_poisson(mu, start, end)
        for mu, start, end
        in zip(mu_v[:n_steps], t_v[:-1], t_v[1:])
    ])


def sim_inhom_clusters(
        lam,
        immigrants=0.0,
        eta=1.0,
        end=np.inf,
        lam_m=None,
        eps=1e-10):
    """
    Vectorised simulation of many inhomogeneous Poisson processes.

    :type lam: function
    :param lam: rate kernel for each cluster,
      a non-negative L_1 integrable function with positive support.

    :type lam_m: function
    :param lam_m: a majorant;
      a non-increasing bounded function which is
      greater than or equal to the lambda function.
      If lambda is non-increasing, it can be its own majorant.

    :type immigrants: numpy.array
    :param immigrants: start time for each process.
      the size of this gives
      the number of independent processes to simulate.
      These will not be returned.

    :type end: numpy.array
    :param end: end time for each process.
      Scalar, or same dimension as start.

    :type eta: float
    :param eta: scale factor for the lam kernel ratio.

    :return: vector of simulated child event times on [T0, T1]. unsorted.
    :rtype: numpy.array

    """

    lam_m = lam_m or getattr(lam, 'majorant', None)
    # Vector of current times
    immigrants = np.asfarray(immigrants)

    substep = 0
    T = immigrants.copy()
    end = end * np.ones_like(T)  # implicit broadcast of possible scalar

    # accumulated times
    allT = [np.array([])]

    while T.size > 0:
        # calculate majorant from current timestep
        if lam_m is not None:
            rate_max = eta * lam_m(T-immigrants)
        else:
            rate_max = lam(T-immigrants)

        # if the majorant has dropped to 0 we are done
        alive = rate_max > eps
        T = T[alive]
        rate_max = rate_max[alive]
        immigrants = immigrants[alive]
        end = end[alive]
        if T.size == 0:
            break

        # simulate for some given rate
        T += random.exponential(1.0/rate_max, T.size)

        # note that the rate we use now is *not* based on the timestep
        # used in the rate majorant but on the incremented times.
        rate = eta * lam(T-immigrants)
        non_spurious = np.random.rand(T.size) * rate_max <= rate
        alive = (T < end)
        # Randomly thin out to correct for majorant and time inc
        allT.append(T[alive & non_spurious].copy())
        T = T[alive]
        immigrants = immigrants[alive]
        end = end[alive]
        substep += 1
    allT = np.concatenate(allT)
    return allT


def sim_branching(
        immigrants,
        phi,
        eta=1.0,
        phi_m=None,
        max_gen=150,  # That's a *lot*
        end=np.inf):
    """
    Simulate Hawkes-type branching process with given immigrants
    Method of Ogata (1981)

    :type phi: function
    :param phi: influence kernel for each event,
      a non-negative function with positive support.
      The L_1 norm of `phi`*`eta` is the branching ratio which,
      if it is greater than 1, will cause explosions, and
      if it is less than 1 will cause cluster extinction and
      if it is equal to 1 will cause overenthusiastic physicists.

    :type eta: function
    :param eta: scale factor for the phi kernel ratio.
      If the L_1 norm of `phi` is 1, this is the branching ratio.

    :type phi_m: function
    :param phi_m: a majorant; a non-increasing L-1 integrable function
      which is greater than or equal to the kernel. If phi is already
      non-increasing it can be its own majorant, and this will be assumed
      as default.

    :type immigrants: np.array
    :param immigrants: An array of seed events; These will not be returned.

    :type max_gen: int
    :param max_gen: Try to stop memory errors by clipping generations

    :type start: float
    :param start: start simulating immigrants at this time

    :type end: float
    :param end: ignore events after this time

    :return: vector of simulated child event times on [T0, T1]. unsorted.
    :rtype: numpy.array
    """

    if phi_m is None:
        phi_m = getattr(phi, 'majorant', phi)
    Tnext = np.asfarray(immigrants)
    allT = [np.array([])]

    for gen in range(max_gen):
        Tnext = sim_inhom_clusters(
            lam=phi,
            eta=eta,
            immigrants=Tnext, end=end,
            lam_m=phi_m)
        # print('gen', gen, Tnext.size)
        if Tnext.size == 0:
            break
        allT.append(Tnext)
    else:
        warn(
            'ran out of generations ({max_gen}) '
            'with {population} still alive'.format(
                max_gen=max_gen,
                population=Tnext.size
            )
        )

    return np.concatenate(allT)


def sim_hawkes(
        phi,
        mu=1.0,
        eta=1.0,
        start=0.0, end=1.0,
        phi_m=None,
        immigrants=None,
        **kwargs):
    """
    Basic Hawkes process

    :type mu: float
    :param mu: background rate of the immigrant process.

    :type immigrants: array
    :param immigrants: external immigrant process

    :type phi: function
    :param phi: influence kernel for each event,
      a non-negative function with positive support.
      The L_1 norm of `phi`*`eta` is the branching ratio which,
      if it is greater than 1, will cause explosions, and
      if it is less than 1 will cause cluster extinction and
      if it is equal to 1 will cause excitable physicists.

    :type eta: function
    :param eta: scale factor for the phi kernel ratio.
      If the L_1 norm of `phi` is 1, this is the branching ratio.

    :type start: float
    :param start: start simulating immigrants at this time

    :type end: float
    :param end: ignore events after this time

    :type phi_m: function
    :param phi_m: a majorant; a non-increasing L-1 integrable function which is
      greater than or equal to the kernel. If phi is already non-increasing it
      can be its own majorant, and this will be assumed as default.

    :return: vector of simulated event times on [T0, T1], unsorted.
    :rtype: numpy.array
    """
    if immigrants is None:
        immigrants = np.array([])
    else:
        immigrants = np.asfarray(immigrants)

    immigrants = np.append(
        immigrants,
        sim_poisson(mu, start, end)
    )
    return np.append(
        immigrants,
        sim_branching(
            immigrants=immigrants,
            phi=phi,
            eta=eta,
            end=end,
            phi_m=phi_m,
            **kwargs
        )
    )
