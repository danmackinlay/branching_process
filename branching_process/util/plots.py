import numpy as np
import matplotlib.pyplot as plt
from . import convert

"""
plots, esp for nonlattice
"""


def piecewise(
        x,
        y,
        *args, **kwargs):
    """
    piecewise plots robust against off-by-1
    """
    x, y = convert.eo(x, y)
    plot = plt.plot(
        x,
        y,
        drawstyle='steps-post',
        *args, **kwargs)
    plt.gca().set_ylim(bottom=0, auto=None)
    return plot


def rug(
        obs_t,
        y=0,
        color="black",
        marker='|',
        ms=20,
        *args, **kwargs):

    plot = plt.scatter(
        obs_t,
        np.ones_like(obs_t) * y,
        marker=marker,
        s=ms,
        c=color,
        *args, **kwargs
    )
    plt.gca().set_ylim(bottom=0, auto=None)
    return plot


def phi_hat_plot(
        model,
        ax=None,
        label='$\hat{\phi}_h(t)$',
        *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = model.basis_lag_.ravel()
    return sparse_step_plot(
        x,
        model.coef_.ravel(),
        ax=ax,
        label=label,
        *args, **kwargs
    )


def phi_weight_plot(
        model,
        ax=None,
        phi_label='$\hat{\phi}_h(t)$',
        label=None,
        scale='log',
        *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.set_yscale(scale)
    x = model.basis_lag_.ravel()
    return sparse_step_plot(
        x,
        model.wlm.penalty_weight.ravel(),
        ax=ax,
        label=label,
        *args, **kwargs
    )


def sparse_step_plot(
        x,
        y,
        ax=None,
        *args, **kwargs):
    """
    piecewise plot of influence kernel
    """
    if ax is None:
        ax = plt.gca()
    last = np.amax(np.nonzero(np.append([1, 1], y)))
    x = x[:last]
    y = y[:last]
    x, y = convert.eo(x, y)
    plot = plt.plot(
        x,
        y,
        drawstyle='steps-post',
        *args, **kwargs)
    ax.set_ylim(0, None, auto=None)
    return plot


def phi_var_plot(
        coef_array,
        phi,
        basis_lag,
        quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],
        label='$\phi$'
        ):
    coef_pc = np.percentile(
        a=coef_array,
        axis=0,
        q=np.array(quantiles) * 100.0
    )
    x, y, y_low, y_high, y_vlow, y_vhigh = convert.e(
        basis_lag,
        coef_pc[2, :],
        coef_pc[1, :],
        coef_pc[-2, :],
        coef_pc[0, :],
        coef_pc[-1, :]
    )

    plt.fill_between(x=x, y1=y_low, y2=y_high, step='post', alpha=0.4)
    plt.fill_between(x=x, y1=y_vlow, y2=y_vhigh, step='post', alpha=0.4)
    plt.plot(x, y, drawstyle='steps-post')

    phi_eval_ts = np.linspace(
        np.amin(basis_lag),
        np.amax(basis_lag),
        100
    )
    phi_eval = phi(phi_eval_ts)
    plt.plot(phi_eval_ts, phi_eval, color="red", label=label)

    return plt.gcf()
