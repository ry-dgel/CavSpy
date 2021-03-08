import numpy as np
import spinmob as sp
import matplotlib.pyplot as plt

import os
plt.style.use(os.path.join(os.path.dirname(__file__),"style.mplstyle"))

from . import data as _d
########
# PSDs #
########
# Generate a PSD from a time trace
def psd_data_file(filename, index=1):
    data = _d.read(filename)
    f, psd = sp.fun.psd(data[0], data[index], window='hanning', rescale=True)
    return f,psd

def psd_data(data, index=1):
    return sp.fun.psd(data[0], data[index], window='hanning', rescale=True)

# Load a PSD file (".A!") from sillyscope
# skip is the length of the header in the file
# idx_offset is added to the y-data column index
# for choosing from multiple channels
def load_file(filename, idx_offset=0, VtoL=None):
    data = _d.read(filename)
    f = data[0]
    y = data[1+idx_offset]
    if VtoL is not None:
        y = y* VtoL**2
    return f,y

# Coarsens a psd exponentially with spinmob,
# level is passed to function
def coarse_psd(x,y,level=1.01):
    return sp._functions.coarsen_data(x,y,level=level,exponential=True)

# Given a PSD file and a plot axis, plots the raw data
# and the coarsened data on top. Returns raw data from file as well.
# linear keyword allows axes to be set to linear, otherwise defaults to loglog
def plot_psd_file(filename, ax, label=None, index=1, VtoL = None, **kwargs):
    if label is None:
        label = filename
    f,y = load_file(filename, index, VtoL)
    plot_psd_data(f,y, ax, **kwargs)

def plot_psd_data(f, y, ax, label=None, level=1.04, linear=None, alpha=0.5, smooth=True, raw=True):
    """Plots the psd contained in the data f,y onto axis ax.

    Parameters
    ----------
    f : np.array
        The data representing the frequency points
    y : np.array
        The data representing the psd points
    ax : matplot.axis
        The axis onto which to plot the PSD
    label : str, optional
        The string to , by default None
    level : float, optional
        [description], by default 1.04
    linear : [type], optional
        [description], by default None
    alpha : float, optional
        [description], by default 0.5
    smooth : bool, optional
        [description], by default True
    raw : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if smooth:
        l = ax.loglog(*coarse_psd(f,y,level=level), label=label, zorder=10)
        if raw:
            ax.loglog(f,y,color=l[0].get_color(),linestyle=':',alpha=alpha,zorder=5)
    elif raw:
        ax.loglog(f,y,label=label,zorder=10)
    if linear is not None:
        if "x" in linear.lower():
            ax.set_xscale("linear")
        if "y" in linear.lower():
            ax.set_yscale("linear")
    return f,y

# Adds a fancy legend above the plot to avoid covering data.
def fancy_leg(fig):
    """Adds a fancy legend to the top of a figure

    Parameters
    ----------
    fig : matplotib.figure
        The figure to add legend to. 
    """
    fig.legend(prop={'size': 8},loc='upper center', 
               bbox_to_anchor=(0.5, 0.9), ncol=4, fancybox=True)

def rms(data):
    """Computes the rms of data.

    Parameters
    ----------
    data : np.array
        The values to compute the rms of

    Returns
    -------
    float
        the rms value of data.
    """
    return np.sqrt(np.mean(np.power(data,2)))

def rms_time(t, y, dt):
    """
    Given evenly spaced time and y data. Splits the data into chunks of width dt
    and calculates the rms of y for each chunk.

    Parameters
    ----------
    t : np.array
        The time data with which to split the corresponding y-data
    y : np.array
        The data to calculate the rms values of.
    dt : float
        The bin width for each section of t to compute the rms of.

    Returns
    -------
    np.array, np.array
        ts contains a list of the average time within each bin. It will have the
        same length as the chunked data.
        rms contains the corresponding rms values for each time bin.
    """
    t_space = np.mean(np.diff(t))
    chunk_size = int(round(dt//t_space))
    print(chunk_size)
    chunked_time = np.resize(t, (t.size//chunk_size,chunk_size))
    chunked_data = np.resize(y, (y.size//chunk_size,chunk_size))
    ts = np.mean(chunked_time,axis=1)
    rms = np.sqrt(np.mean(np.power(chunked_data,2),axis=1))
    return ts, rms
