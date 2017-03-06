try:
    import autograd.numpy as np
    import autograd.scipy as sp
    have_autograd = True
except ImportError as e:
    import numpy as np
    import scipy as sp
    have_autograd = False



def intensity_hawkes(
        timestamps,
        mu,
        phi,
        eta=1.0,
        eval_timestamps=None,
        sort=True,
        max_floats=1e8,
        **kwargs):
    """
    True intensity of Hawkes process.
    Memory-hungry; could be improved, with numba.
    """
    timestamps = np.asarray(timestamps).ravel()
    if sort:
        timestamps = np.sort(timestamps)
    if eval_timestamps is None:
        eval_timestamps = timestamps
        if sort:
            eval_timestamps = np.sort(eval_timestamps)
    eval_timestamps = np.asarray(eval_timestamps).ravel()
    if ((timestamps.size) * (eval_timestamps.size)) > max_floats:
        return _intensity_hawkes_lite(
            timestamps=timestamps,
            mu=mu,
            phi=phi,
            eta=eta,
            eval_timestamps=eval_timestamps,
            **kwargs)
    deltas = eval_timestamps.reshape(1, -1) - timestamps.reshape(-1, 1)
    mask = deltas > 0.0
    endo = phi(deltas) * mask
    lambdas = endo.sum(0) * eta + mu
    return lambdas


def _intensity_hawkes_lite(
        timestamps,
        eval_timestamps,
        mu,
        phi,
        eta=1.0,
        **kwargs):
    """
    True intensity of hawkes process.
    Memory-lite version. CPU-hungry, could be improved with numba.
    """
    endo = np.zeros_like(eval_timestamps)
    deltas = np.zeros_like(timestamps)
    mask = np.zeros_like(timestamps)
    for i in range(eval_timestamps.size):
        deltas[:] = eval_timestamps[i] - timestamps
        mask[:] = deltas > 0.0
        endo[i] = np.sum(phi(deltas) * mask)
    return endo * eta + mu
