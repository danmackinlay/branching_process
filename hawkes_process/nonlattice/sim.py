"""
Converting cts time data sets to discrete time
"""

import numpy as np
from math import floor, ceil
from . import sim_cts
from . import convert


def quantize_timestamps(
        timestamps,
        obs_t=None,
        cumulative=True,
        *args, **kwargs):
    if obs_t is None or obs_t == 'equal':
        obs_t = obs_times_equal(timestamps, *args, **kwargs)
    elif obs_t == 'poisson':
        obs_t = obs_times_poisson(timestamps, *args, **kwargs)
    counts, obs_t = np.histogram(timestamps, obs_t)
    start_count = np.sum(timestamps < obs_t[0])
    if cumulative:
        obs_t, counts = convert.interval_counts_as_cumulative_counts(
            obs_t, counts, start_count=start_count
        )
    return obs_t, counts


def obs_times_equal(
        timestamps,
        rate=1.0,
        endpoints=False,
        eps=1e-9,
        *args, **kwargs):
    """
    sensible equal sample times for a given timestamp list
    """
    scaled_timestamps = timestamps/rate
    start_i = floor(np.amin(scaled_timestamps) - eps)
    end_i = ceil(np.amax(scaled_timestamps) + eps)
    n = end_i - start_i + 1
    times = np.linspace(start_i, end_i, n, endpoint=True) * rate
    if endpoints:
        times[0] = timestamps[0]
        times[-1] = timestamps[-1]
        return times
    else:
        return times[1:-1]


def obs_times_poisson(
        timestamps,
        rate=1.0,
        endpoints=False,
        *args, **kwargs):
    """
    sensible random sample times for a given timestamp list

    :type rate: float
    :param rate: rate of the sampling process.

    :type endpoints: Boolean
    :param endpoints: include, exclude, or extrapolate the endpoints

    :return: sorted vector of interval times
    :rtype: numpy.array
    """
    seq_start_t = np.amin(timestamps)
    seq_end_t = np.amax(timestamps)
    times = np.sort(
        sim_cts.sim_poisson(
            start=seq_start_t,
            end=seq_end_t,
            mu=rate
        )
    )

    if endpoints:
        return np.concatenate(([seq_start_t], times, [seq_end_t]))
    return times
