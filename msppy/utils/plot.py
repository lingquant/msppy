#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import matplotlib.pyplot as plt
import numpy

def fan_plot(x, ax=None):
    """Create fan_plot of a given DataFrame

    Parameters
    ----------
    x: bidimensional numpy array (n_samples*T)

    ax: Matplotlib AxesSubplot instance, optional
        The specified subplot is used to plot; otherwise a new figure is created.

    Returns
    -------
    matplotlib.pyplot.figure instance
    """
    if x.ndim != 2:
        raise Exception("input of x is not valid!")
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    avg = numpy.mean(x, axis=0)
    ax.plot(avg,'--',color='black')
    for sample in x:
        ax.plot(sample,alpha=0.05,color='purple')
        ax.fill_between(
            range(x.shape[1]),
            sample,
            avg,
            alpha=0.05,
            color='purple'
        )
    return fig

def plot_bounds(db, pv, sense=1, percentile=95, start=0, window=1, smooth=0, ax=None):
    """plot the evolution of bounds

    Parameters
    ----------
    db: unidimensional array-like
        An T-length array of the determinstic bounds

    pv: bidimensional array-like
        An (n_iterations*n_steps) array of the policy values

    sense: -1/1 (default=1)
        The modelsense: 1 indicates min problem and 1 indicates max problem.

    percentile: float (default=95)
        The percentile used to construct confidence interval.

    ax: Matplotlib AxesSubplot instance, optional
        The specified subplot is used to plot; otherwise a new figure is created.

    window: int, optional (default=1)
        The length of the moving windows to aggregate the policy values. If
        length is bigger than 1, approximate confidence interval of the
        policy values and statistical bounds will be plotted.

    smooth: bool, optional (default=0)
        If 1, fit a smooth line to the policy values to better visualize
        the trend of statistical values/bounds.

    start: int, optional (default=0)
        The start iteration to plot the bounds. Set start to other values
        can zoom in the evolution of bounds in most recent iterations.

    Returns
    -------
    matplotlib.pyplot.figure instance
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    from matplotlib.ticker import MaxNLocator
    from msppy.utils.statistics import compute_CI
    if smooth == 1:
        from msppy.utils.statistics import fit
    db = numpy.array(db)
    pv = numpy.array(pv)
    end = len(db)
    n_processes = pv.shape[1]
    x_value = range(start,end)
    ax.plot(
        x_value,
        db[start:end],
        '-b',
        label = 'deterministic bounds'
    )
    pv_unpack = [item for alist in pv for item in alist]
    if n_processes != 1 or window != 1:
        x_value = range(max(start,window-1),end)
        CI = [
            compute_CI(
                pv_unpack[n_processes*(i-window+1):n_processes*(i+1)],
                percentile,
            )
            for i in range(window-1,end)
        ]
        CI = CI[max(start,window-1)-window+1:end]
        CI_lower_end = [item[0] for item in CI]
        CI_upper_end = [item[1] for item in CI]
        CI_mid = [sum(item)/len(item) for item in CI]
        ax.fill_between(
            x_value,
            CI_lower_end,
            CI_upper_end,
            facecolor='pink',
            alpha=0.5,
            edgecolor='none',
            label='expected policy values {}% CI'.format(percentile)
        )
        if sense == 1:
            ax.plot(
                x_value,
                CI_upper_end,
                '-r',
                label='statistical bounds '+str(percentile)+'% C'
            )
            if smooth == 1:
                ax.plot(
                    x_value,
                    fit(CI_mid, convex=1),
                    '--g',
                    label='smoothed policy values'
                )
        else:
            ax.plot(
                x_value,
                CI_lower_end,
                '-r',
                label='statistical bounds '+str(percentile)+'% C'
            )
            if smooth == 1:
                ax.plot(
                    x_value,
                    fit(CI_mid, convex=-1),
                    '--g',
                    label='smoothed policy values'
                )
    else:
        pv = pv[start:end]
        pv = [item[0] for item in pv]
        ax.plot(
            x_value,
            pv,
            '-r',
            label='policy values'
        )
        if smooth == 1:
            ax.plot(
                x_value,
                fit(pv,sense),
                '--g',
                label='smoothed policy values'
            )
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Values')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc = 'best')
    ax.set_title('Evolution of bounds')
    return fig
