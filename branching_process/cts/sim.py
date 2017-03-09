import numpy as np
from numpy import random
from warnings import warn
from . import influence


def sim_poisson(
        mu,
        t_start=0.0,
        t_end=1.0):
    """
    Simulate constant-rate Poisson process.

    :type mu: float
    :param mu: rate of the process.

    :type t_start: float
    :param t_start: t_start simulating immigrants at this time

    :type t_end: float
    :param t_end: no events after this time

    :return: vector of simulated event times on [t_start, t_end], unsorted.
    :rtype: numpy.array
    """
    timespan = t_end-t_start
    N = random.poisson(lam=mu*timespan)
    return t_start + random.rand(N)*timespan


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
        sim_poisson(mu, t_start, t_end)
        for mu, t_start, t_end
        in zip(mu_v[:n_steps], t_v[:-1], t_v[1:])
    ])


def sim_inhom_clusters(
        lam,
        immigrants=0.0,
        eta=1.0,
        t_end=np.inf,
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
    :param immigrants: t_start time for each process.
      the size of this gives
      the number of independent processes to simulate.
      These will not be returned.

    :type t_end: numpy.array
    :param t_end: t_end time for each process.
      Scalar, or same dimension as t_start.

    :type eta: float
    :param eta: scale factor for the lam kernel ratio.

    :return: vector of simulated child event times on [T0, T1]. unsorted.
    :rtype: numpy.array

    """
    lam = influence.as_influence_kernel(lam, majorant=lam_m)
    # Vector of current times
    immigrants = np.asfarray(immigrants)

    substep = 0
    T = immigrants.copy()
    t_end = t_end * np.ones_like(T)  # implicit broadcast of possible scalar

    # accumulated times
    allT = [np.array([])]

    while T.size > 0:
        # calculate majorant from current timestep
        rate_max = eta * lam.majorant(T-immigrants)

        # if the majorant has dropped to 0 we are done
        alive = rate_max > eps
        T = T[alive]
        rate_max = rate_max[alive]
        immigrants = immigrants[alive]
        t_end = t_end[alive]
        if T.size == 0:
            break

        # simulate for some given rate
        T += random.exponential(1.0/rate_max, T.size)

        # note that the rate we use now is *not* based on the timestep
        # used in the rate majorant but on the incremented times.
        rate = eta * lam(T-immigrants)
        non_spurious = np.random.rand(T.size) * rate_max <= rate
        alive = (T < t_end)
        # Randomly thin out to correct for majorant and time inc
        allT.append(T[alive & non_spurious].copy())
        T = T[alive]
        immigrants = immigrants[alive]
        t_end = t_end[alive]
        substep += 1
    allT = np.concatenate(allT)
    return allT


def sim_branching(
        immigrants,
        phi_kernel,
        eta=1.0,
        phi_m=None,
        max_gen=150,  # That's a *lot*
        t_end=np.inf):
    """
    Simulate Hawkes-type branching process with given immigrants
    Method of Ogata (1981)

    :type phi_kernel: function
    :param phi_kernel: influence kernel for each event,
      a non-negative function with positive support.
      The L_1 norm of `phi_kernel`*`eta` is the branching ratio which,
      if it is greater than 1, will cause explosions, and
      if it is less than 1 will cause cluster extinction and
      if it is equal to 1 will cause overenthusiastic physicists.

    :type eta: function
    :param eta: scale factor for the phi_kernel kernel ratio.
      If the L_1 norm of `phi_kernel` is 1, this is the branching ratio.

    :type phi_m: function
    :param phi_m: a majorant; a non-increasing L-1 integrable function
      which is greater than or equal to the kernel. If phi_kernel is already
      non-increasing it can be its own majorant, and this will be assumed
      as default.

    :type immigrants: np.array
    :param immigrants: An array of seed events; These will not be returned.

    :type max_gen: int
    :param max_gen: Try to stop memory errors by clipping generations

    :type t_start: float
    :param t_start: t_start simulating immigrants at this time

    :type t_end: float
    :param t_end: ignore events after this time

    :return: vector of simulated child event times on [T0, T1]. unsorted.
    :rtype: numpy.array
    """

    phi_kernel = influence.as_influence_kernel(phi_kernel, majorant=phi_m)
    Tnext = np.asfarray(immigrants)
    allT = [np.array([])]

    for gen in range(max_gen):
        Tnext = sim_inhom_clusters(
            lam=phi_kernel,
            eta=eta,
            immigrants=Tnext,
            t_end=t_end,)
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
        phi_kernel,
        mu=1.0,
        eta=1.0,
        t_start=0.0, t_end=1.0,
        phi_m=None,
        immigrants=None,
        sort=True,
        **kwargs):
    """
    Basic Hawkes process

    :type mu: float
    :param mu: background rate of the immigrant process.

    :type immigrants: array
    :param immigrants: external immigrant process

    :type phi_kernel: function
    :param phi_kernel: influence kernel for each event,
      a non-negative function with positive support.
      The L_1 norm of `phi_kernel`*`eta` is the branching ratio which,
      if it is greater than 1, will cause explosions, and
      if it is less than 1 will cause cluster extinction and
      if it is equal to 1 will cause excitable physicists.

    :type eta: function
    :param eta: scale factor for the phi_kernel kernel ratio.
      If the L_1 norm of `phi_kernel` is 1, this is the branching ratio.

    :type t_start: float
    :param t_start: t_start simulating immigrants at this time

    :type t_end: float
    :param t_end: ignore events after this time

    :type phi_m: function
    :param phi_m: a majorant; a non-increasing L-1 integrable function which is
      greater than or equal to the kernel. If phi_kernel is already non-increasing it
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
        sim_poisson(mu, t_start, t_end)
    )
    ts = np.append(
        immigrants,
        sim_branching(
            immigrants=immigrants,
            phi_kernel=phi_kernel,
            eta=eta,
            t_end=t_end,
            phi_m=phi_m,
            **kwargs
        )
    )
    if sort:
        ts = np.sort(ts)
    return ts
