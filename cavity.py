import functools
from os import linesep
import numpy as np
from numba import jit
from scipy import constants
from scipy.signal import find_peaks
from multiprocessing import Pool
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
#TODO: Implement fitting with lmfit
import lmfit as lm

from . import data as _d
from . import uncert as _u

# Pretty Plotting
import os
plt.style.use(os.path.join(os.path.dirname(__file__),"style.mplstyle"))

pi = constants.pi
c  = constants.c

##################
# Analytic Error #
##################
@jit(nopython=True)
def errf(dL, r, fm, m, lamb, a, phi, theta):
    dl = (4 * pi / lamb) * dL
    lm = 2 * pi * fm * m * lamb / c
    poly1=(24 - 12*dl**2 + dl**4)
    t1 = ((lm**2 *
          (-24 + lm**2)**2 * r**2 * (-1 + r**2) * (3 * dl * r**2 *
            (-1 + r**2) * (6 *
                (-3 + lm**2 + 2 * r**2 + r**4) -
                dl**2 * (-3 + lm**2 + 8 * r**2 + r**4)) +
            3 * a * dl * (-6 + dl**2 + 4 *
                (-3 + 2 * dl**2) * r**2 + 2 * (-6 + dl**2) * (-3 + lm**2) * r**4 +
                4 * (-3 + 2 * dl**2) * r**6 + (-6 + dl**2) * r**8) * np.cos(phi) -
            a * (-1 + r**4) * (9 *
                (-2 + dl**2) + (36 + dl**2 * (-6 + dl**2)**2 - 18 * lm**2) * r**2 +
                9 * (-2 + dl**2) * r**4)  * np.sin(phi))) /
          (32. * (1 + (-2 + dl**2) * r**2 + r**4) * (18 + r**2 * (6 * dl * lm * (-6 + lm**2) - dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2))) *
          (18 + r**2 * (-6 * dl * lm * (-6 + lm**2) + dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2)))))

    t2 = ((-7776 * (lm - lm**3 / 6.) * r**2 * (-1 + r**2) * ((2 * dl - (4 * dl**3) / 3.) * r**2 * (r**4 + a * np.cos(phi)) +
         (dl - dl**3 / 6.) * ((-2 + lm**2) * r**2 * (-1 + r**2) - (poly1 * (r**4 + a * r**6 * np.cos(phi))) / 12. + (-1 + r**4) * (r**2 - r**4 + a * (1 + r**4) * np.cos(phi))) -
         a * (dl - dl**3 / 6.)**2 * r**2 * (1 + r**4) * np.sin(phi) + a * (-((-24 + 12 * dl**2 - dl**4 + 24 * r**2) * (-24 + poly1 * r**2) * (1 + r**4)) / 576. +
         2 * (1 - lm**2 / 2.) * (r**2 + (-2 + dl**2 - dl**4 / 12.) * r**4 + r**6)) * np.sin(phi))) /
         ((12 - poly1 * r**2 + 12 * r**4) * (18 + r**2 * (6 * dl * lm * (-6 + lm**2) - dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2))) *
         (18 + r**2 * (-6 * dl * lm * (-6 + lm**2) + dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2)))))

    err = t1*np.cos(theta) + t2*np.sin(theta)
    return err

#########################
# Analytic Transmission #
#########################
@jit(nopython=True)
def transf(dL, r, fm, m, lamb, pc):
    dl = (4 * pi / lamb) * dL
    lm = 2 * pi * fm * m * lamb / c
    poly1= (24 - 12*dl**2 + dl**4)
    # Carrier Trans
    cr = (12*pc*(-1 + r**2)**2)/(12 - poly1*r**2 + 12*r**4)
    # Sideband Trans
    sb = (-((-1 + pc)*(-1 + r**2)**2 * (288 - poly1*(24 - 12*lm**2 + lm**4)*r**2 + 288*r**4))/
           (72.*(2 - (4 - 2*dl**2 + 4*dl*lm + (-2 + dl**2)*lm**2)*r**2 + 2*r**4) *
                (2 + r**2*(4*dl*lm - dl**2*(-2 + lm**2) + 2*(-2 + lm**2 + r**2)))))
    tr = cr+sb
    return tr

#############
# Normalize #
#############
@jit(nopython=True)
def norm(vals):
    return vals/np.max(np.abs(vals))

###########################
# Parametric Minimization #
###########################
@jit(nopython=True, parallel=True) # We want this to be as fast as possible, so let's JIT it with parallelization
def min_dist(pair,es,ts,sigmae,sigmat):
    e = pair[0]
    t = pair[1]
    return np.min(np.hypot((e-es)/sigmae,(t-ts)/sigmat))

def min_dists(pairs,es,ts,sigmae,sigmat):
    func = functools.partial(min_dist, es=es, ts=ts,sigmae=sigmae,sigmat=sigmat)
    with Pool(10) as p:
        return np.array(p.map(func, pairs))

#########################
# Simple Fitting Funcs. #
#########################
def fano(x, amp, slope, width, center):
    hw = width/2
    deltax = x-center
    return (amp + slope * deltax) * hw**2 /((hw)**2 + deltax**2)

def triple_fan(x, splitting, amp, slope, center, linewidth, ps, offset):
    shift = splitting/2
    carrier = fano(x, amp, slope, linewidth, center)
    sidebands = ps * (fano(x, amp, slope, linewidth, center+shift) 
                      + fano(x, amp, slope, linewidth, center-shift))
    return offset + carrier + sidebands

def lorenz(x, amp, width, center):
    hw = width/2
    deltax = x-center
    return (amp) * hw**2 /((hw)**2 + deltax**2)

def triple_lor(x, splitting, amp, center, linewidth, ps, offset):
    shift = splitting/2
    carrier = lorenz(x, amp, linewidth, center)
    sidebands = ps * (lorenz(x, amp, linewidth, center+shift) 
                      + lorenz(x, amp, linewidth, center-shift))
    return offset + carrier + sidebands

####################
# Sideband Fitting #
####################
def fit_triple(filename, func, mod_freq, ax=None, idx_offset=0, sb_ratio=10, lw_ratio=2):
    print("Fitting sideband data in %s" % filename)
    data = _d.read(filename)
    xs = data[0]
    ys = data[1+idx_offset]

    # Getting main peak
    peak=find_peaks(ys, height=(np.mean(ys)+0.1*np.std(ys)),distance=50000)[0][0]

    # Rough guessing side peaks
    lw_left = np.argmin(np.abs(ys[:peak] - np.max(ys[:peak])/lw_ratio))
    lw_right = len(xs) - np.argmin(np.abs(np.flip(ys[peak:]) - np.max(ys[peak:])/lw_ratio))
    split_left = np.argmin(np.abs(ys[:peak] - np.max(ys[:peak])/sb_ratio))
    split_right = len(xs) - np.argmin(np.abs(np.flip(ys[peak:]) - np.max(ys[peak:])/sb_ratio))

    # Guesses
    offset = np.mean(ys)/2
    center = xs[peak]
    split = (xs[split_right]-xs[split_left])
    amp = max(ys) - offset
    sigma = min(np.diff(ys))
    lw = (xs[lw_right] - xs[lw_left])

    # Fitting
    model = lm.Model(func)
    params = model.make_params(splitting=split, 
                               amp=amp, 
                               center=center, 
                               linewidth=lw, 
                               ps=1/sb_ratio, 
                               offset=offset,
                               slope=0.0001)
    
    # Computing results
    result = model.fit(ys, params, x=xs, weights = 1/sigma * np.ones(ys.size))
    chisqr = result.redchi
    best_vals = result.best_values
    if ax is not None:
        ax.plot(xs, func(xs,*(best_vals.values())))
    if chisqr > 1.5:
        print("Chi-Square from triplet fit is greater than 1.5!")
        return None
    best_vals = _u.from_fit(result)
    # Splitting is the distance between sidebands, and so total
    # Frequency difference is twice the modulation frequency.
    lw = best_vals['linewidth'] / best_vals['splitting'] * (2 * mod_freq)
    print("Linwidth = %.2f MHz (Chisq = %.2f)" % (np.abs(lw),chisqr))
    return np.abs(lw)

#####################
# WhiteLight Length #
#####################
def white_length(filename, plot=False, disp=False, col=10,
                 wlmin=600.0, wlmax=650.0, dist=50, height=None, ratio=0.05, **kwargs):
    """
    Fit the whitelight data present in the given file.
    This assumes that the data was bined and only contained in one row of the
    saved data.


    Parameters
    ----------
    filename : string
        the file containing the counts data.
    plot : bool/matplotlib.axes, optional
        If True, will generate a plot of the data, by default False.
        Can also pass a plot axes object to plot directly on that plot.
    disp : bool, optional
        If True, will print the results, by default False
    col : Int
        Which column of the data to use optional, by default 10
    wlmin : float, optional
        The minimum wavelength to include in the peak detect region, by default 600.0
    wlmax : float, optional
        The maxmimum wavelength to include in the peak detect region, by default 650.0
    dist : int, optional
        The minimum distance between x-points to consider new peaks, by default 50
    height : int, optional
        The minimum value above which a peak must reach to be considered, if none
        will be ratio times the max count number, plus the mean count number.
    ratio : float, optional
        How small of a bump relative to the max to consider a peak, by default 0.05
    **kwargs will be passed to find_peaks

    Returns
    -------
    float, float
        The computer length in um, and FSR in MHz
    """
    wl_data = _d.read_csv(filename, names=['wl','counts'],df=True,usecols=[0,10],head=0,skiprows=30,delim=',')
    wavelength = wl_data['wl'].to_numpy()
    counts = wl_data['counts'].to_numpy()
    if height is None:
        height = np.mean(counts) + (np.max(counts) * ratio)

    bounds = np.logical_and(wavelength >= wlmin, wavelength <= wlmax)
    wavelength = wavelength[bounds]
    counts = counts[bounds]

    peaks = find_peaks(counts, height=height, distance=dist, **kwargs)
    peak_wl = wavelength[peaks[0]]
    peak_freq = c/(peak_wl * 1E-9)
    fsrs = np.diff(peak_freq[::-1])
    lengths = c/(2*fsrs)

    length = _u.from_floats(lengths * 1E6)  # um
    fsr = _u.from_floats(fsrs / 1E6) # MHz

    if plot == True:
        plt.figure()
        plt.plot(wavelength,counts)
        plt.vlines(peak_wl,np.min(counts),np.max(counts)+10)
        plt.show(block=False)
    elif plot != False:
        plot.plot(wavelength, counts)
        plot.vlines(peak_wl,np.min(counts),np.max(counts)+10)
    if disp:
        print(filename)
        print("\tCavity length is: %s um" % length)
        print("\tFSR is: %s MHz" % fsr)
    return length, fsr
