import numpy as np

"""
data format conversions, esp for nonlattice
"""


def interval_counts_as_rates(obs_t, interval_counts):
    interval_counts = np.asarray(interval_counts)
    rates = interval_counts / obs_times_to_delta_times(obs_t)
    return obs_t, rates


def rates_as_interval_counts(rates, obs_t, round=True):
    obs_t = np.asarray(obs_t)
    rates = np.asarray(rates)
    interval_counts = rates * obs_times_to_delta_times(
        obs_t
    )
    if round:
        interval_counts = np.around(interval_counts)
    return obs_t, interval_counts


def cumulative_counts_as_rates(obs_t, cum_obs):
    return interval_counts_as_rates(
        *cumulative_counts_as_interval_counts(
            obs_t, cum_obs
        )
    )


def cumulative_counts_as_interval_counts(obs_t, cum_obs):
    obs_t = np.asarray(obs_t)
    cum_obs = np.asarray(cum_obs)
    sample_counts = np.diff(cum_obs)
    return obs_t, sample_counts


def interval_counts_as_cumulative_counts(
        obs_t,
        interval_counts,
        start_count=None):
    obs_t = np.asarray(obs_t)
    cum_obs = np.cumsum(interval_counts)
    if start_count is not None:
        cum_obs = np.concatenate((
            [start_count],
            cum_obs
        ))
    return obs_t, cum_obs


def rates_as_cumulative_counts(obs_t, rates, start_count=None):
    obs_t, sample_counts = rates_as_interval_counts(obs_t, rates)
    return interval_counts_as_cumulative_counts(
        obs_t, sample_counts, start_count=start_count
    )


def obs_times_to_delta_times(obs_t):
    obs_t = np.asarray(obs_t)
    return np.diff(obs_t)


def tt(obs_time, *whatevers):
    """
    trim time; cuts off the extra timestamps as needed
    """
    n = whatevers[0].size
    return tuple(
        [obs_time[:n], ] +
        list(whatevers)
    )


def eo(obs_time, *whatevers):
    """
    extend obs; repeats a sample count if necessary to get even step plots
    """
    obs_time = np.asarray(obs_time)
    n_time = obs_time.size
    new_whatevers = []
    for whatever in whatevers:
        whatever = np.asarray(whatever)
        n_obs = whatever.size
        if n_obs < n_time:
            whatever = np.concatenate((
                whatever,
                [whatever[-1]] * (n_time-n_obs)
            ))
        new_whatevers.append(whatever)
    return tuple([obs_time] + new_whatevers)


def et(obs_time, *whatevers):
    """
    extend time; repeats a sample timestep if necessary
    to get even step plots
    This really only makes sense if we only add one step to the array.
    """
    obs_time = np.asarray(obs_time)
    n_time = obs_time.size
    n_obs = np.amax([w.size for w in whatevers])
    if n_obs > n_time:
        delta_t = obs_time[-1] - obs_time[-2]
        shortfall = n_obs-n_time
        obs_time = np.concatenate((
            obs_time,
            np.linspace(
                obs_time[-1] + delta_t,
                obs_time[-1] + delta_t * shortfall,
                shortfall,
                endpoint=True
            ),
        ))

    return tuple((obs_time,) + whatevers)


def e(obs_time, *whatevers):
    """
    extend what is necessary to extend.
    """
    return et(*eo(obs_time, *whatevers))
