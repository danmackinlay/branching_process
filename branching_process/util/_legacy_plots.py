#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import _settings
import os.path
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import colors, ticker, cm
from math import log10

def logspace_for(data, n=20):
    """log10-spaced bins for some data"""
    data = np.asarray(data)
    top = data.max()
    bot = data.min()
    logtop = log10(top)
    logbot = log10(bot)
    logspace = np.linspace(logbot, logtop, n, endpoint=True)
    space = 10.0 ** logspace
    space[0]=bot
    space[-1]=top
    return space

def contour_cum_histo(x, y,
        title="",
        xlabel="",
        ylabel="",
        n_bins=50,
        n_levels=10,
        x_cap_pc=100.0,
        y_cap_pc=100.0,
        cmap=None):
    "histogram with approx equal-occupancy contour lines"
    if cmap is None: cmap = plt.cm.bone_r
    x_cap = np.percentile(x, x_cap_pc)
    y_cap = np.percentile(y, y_cap_pc)
    mask = (x<=x_cap) & (y<=y_cap)
    x_capped = x[mask]
    y_capped = y[mask]

    H, xedges, yedges = np.histogram2d(
        x_capped, y_capped,
        bins=(n_bins, n_bins),
        normed=True)
    H_sorted = np.sort(H.flatten())
    H_cum = H_sorted.cumsum()
    # more precise version at https://gist.github.com/adrn/3993992
    levels = H_sorted[H_cum.searchsorted(np.linspace(1.0/n_levels*H_cum[-1], H_cum[-1], n_levels, endpoint=True))]
    level_labels = np.linspace(0, 100.0*(1-1.0/n_levels), n_levels, endpoint=True)
    #lowest_bin =  np.percentile(H[H>0].flatten(), 5.0) #Ignore bottom 5%
    #levels = np.power(10,np.arange(np.ceil(np.log(lowest_bin)),np.ceil(np.log(H.max())), 0.5))
    #levels = np.concatenate([[0.0], levels])
    #levels = np.percentile(H.flatten(), np.linspace(0.0, 100.0, n_levels, endpoint=True))
    #extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] #axes transposed for histograms
    fig = plt.figure()
    ax = plt.gca()

    #points = plt.scatter(
    #    y_capped, x_capped,
    #                     marker="x"
    #)

    cset = plt.contourf(H,
        levels=levels,
        cmap=cmap,
        #origin='lower',
        #colors=['black','green','blue','red'],
        #locator=ticker.LogLocator(),
        #linewidths=(1.9, 1.6, 1.5, 1.4),
        extent=extent
    )
    fset = plt.contour(H,
        levels=levels,
        #origin='lower',
        colors=['red'],
        #locator=ticker.LogLocator(),
        #linewidths=(1.9, 1.6, 1.5, 1.4),
        extent=extent,
        hold='on'
    )
    # Make a colorbar for the ContourSet returned by the contourf call.
    #cbar = plt.colorbar(cset)
    #cbar.ax.set_ylabel('verbosity coefficient')
    # Add the contour line levels to the colorbar
    #cbar.add_lines(fset)
    #plt.clabel(cset, inline=1, fontsize=10, fmt='%1.0i')
    #for c in cset.collections:
    #    c.set_linestyle(‘solid’)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax, cset, fset #, cbar

def counts_for(timestamps):
    "small helper to create an index vector"
    return np.arange(1, timestamps.size+1)

def plot_timestamps(timestamps, ax=None, **kwargs):
    ax = ax if ax else plt.gca()
    return plt.plot(timestamps, counts_for(timestamps), **kwargs)

def plot_point_ts_series(tseries, ax=None, **kwargs):
    ax = ax if ax else plt.gca()
    return plt.plot(tseries.index, tseries, **kwargs)

def plot_ts(ts_frame, ax=None, **kwargs):
    ax = ax if ax else plt.gca()
    return ax.plot(ts_frame.run_time, ts_frame.view_count, **kwargs)

def multisave(basename, fig=None, dpi=300, **kwargs):
    basedir = getattr(_settings, 'FIGURES', None)
    fig = fig if fig else plt.gcf()
    if basedir:
        basename = os.path.join(basedir, basename)
    #Aggressively prevent file handle leakage
    with open(basename + ".png", "w") as h:
        fig.savefig(h, format="png", dpi=dpi)
    with open(basename + ".pdf", "w") as h:
        fig.savefig(h, format="pdf")
    with open(basename + ".svg", "w") as h:
        fig.savefig(h, format="svg")
    #return fig

def plot_ts_rates(ts_frame, ax=None,
        title=None,
        scale=3600*24, **kwargs):
    ax = ax if ax else plt.gca()
    vid = ts_frame.iloc[0,0]
    if title is None:
        title = "Estimated rate for {!r}".format(vid)
    ax.step(
        pd.to_datetime(ts_frame.run_time[1:] * scale, unit='s'),
        ts_frame.rate[1:],
        **kwargs)
    #ax.set_xlabel('time')
    ax.set_ylabel('approx. intensity (views/day)')
    ax.set_title(title)
    ax.figure.autofmt_xdate()
    return ax

def diagnose_ts(ts_frame, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    ax = axes[0]
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

    fig.tight_layout()
    return fig, axes
