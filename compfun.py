import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from . import data

# Pretty Plotting
import os
plt.style.use(os.path.join(os.path.dirname(__file__),"style.mplstyle"))

###############################;
# Numerical Complex Functions #
###############################
class CompFun:
    def __init__(self, comp, freq):
        if not len(comp) == len(freq):
            raise ValueError("Length Mismatch")

        self.c = np.asarray(comp)
        self.f = np.asarray(freq)

    @classmethod
    def from_polar(cls, polar):
        return cls(polar['r'] * np.exp(1j * polar['phase']),
                       polar['frequency'])

    @classmethod
    def from_cart(cls, cartesian):
        return cls(cartesian['x'] + 1.0j * cartesian['y'],
                   cartesian['frequency'])

    def __eq__(self, other):
        return (np.array_equal(self.c, other.c)
                and np.array_equal(self.f, other.f))

    def __add__(self,other):
        if isinstance(other, CompFun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return CompFun(self.c + other.c, self.f)
        else:
            return CompFun(self.c + other, self.f)

    def __neg__(self):
        return CompFun(-self.c, self.f)

    def __sub__(self,other):
        if isinstance(other, CompFun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return CompFun(self.c - other.c, self.f)
        else:
            return CompFun(self.c - other, self.f)

    def __mul__(self, other):
        if isinstance(other, AnCompFun):
            return CompFun(self.c * other.func(self.f), self.f)

        elif isinstance(other, CompFun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return CompFun(self.c * other.c, self.f)
        elif isinstance(other, float):
            return CompFun(self.c * other, self.f)
    def __truediv__(self,other):

        if isinstance(other, AnCompFun):
            return CompFun(self.c / other.func(self.f), self.f)
        elif isinstance(other, CompFun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mistmatch between two functions")
            return CompFun(self.c / other.c, self.f)
        elif isinstance(other, float):
            return CompFun(self.c / other, self.f)

    def polar(self):
        return {'r': np.abs(self.c),
                'phase': np.angle(self.c),
                'frequency': self.f}

    def plot(self, **kwargs):
        return plot_trans(self.polar(), **kwargs)
    
    def save(self, filename, polar=False):
        if polar:
            dat = self.polar()
            np.savetxt(filename,np.transpose(np.array([dat['frequency'],dat['r'],dat['phase']])),delimiter=',')
        else:
            np.savetxt(filename,np.transpose(np.array([self.f,np.real(self.c),np.imag(self.c)])))

# Combines the dataset of two CompFun objects.
# Averages data points of identical frequency, concats other points.
def merge(cm1, cm2):
    c = cm1.c
    f = cm1.f
    newfs = []
    newcs = []
    for i, freq in enumerate(cm2.f):
        if freq in f:
            j = np.argmin(np.abs(f - freq))
            c[j] = np.average([cm1.c[j], cm2.c[i]])
        else:
            newfs.append(freq)
            newcs.append(cm2.c[i])
    f = np.append(f, np.array(newfs))
    c = np.append(c, np.array(newcs))
    combined = zip(c,f)
    combined = sorted(combined, key = lambda pair: pair[1])
    c,f = map(list, zip(*combined))
    return CompFun(c, f)

def load(filename, polar=True):
    if polar:
        return [CompFun.from_polar(chunk) for chunk in
                data.unpack(filename, fields=['r','phase','frequency'],delim=',')]
    else:
        return [CompFun.from_cart(chunk) for chunk in
                data.unpack(filename, fields=['x','y','frequency'],delim=',')]

##############################
# Analytic Complex Functions #
##############################
class AnCompFun:
    def __init__(self, function):
        self.func = function

    def __add__(self, other):
        if isinstance(other, AnCompFun):
            return AnCompFun(lambda f: self.func(f) + other.func(f))
        elif isinstance(other, CompFun):
            return CompFun(self.func(other.f) + other.c, other.f)

    def __mul__(self, other):

        if isinstance(other, CompFun):
            return CompFun(self.func(other.f) * other.c, other.f)

        elif isinstance(other, AnCompFun):
            return AnCompFun(lambda f: (self.func(f) * other.func(f)))

        elif isinstance(other, float):
            return AnCompFun(lambda f: (self.func(f) * other))

    def __truediv__(self, other):
        if isinstance(other, CompFun):
            return CompFun(self.func(other.f) / other.c, other.f)

        elif isinstance(other, AnCompFun):
            return AnCompFun(lambda f: (self.func(f) / other.func(f)))

        elif isinstance(other, float):
            return AnCompFun(lambda f: (self.func(f) / other))

    def apply(self, freq):
        return CompFun(self.func(freq), freq)

    def plot(self,freq, **kwargs):
        return self.apply(freq).plot(**kwargs)

# High Pass Filter with cutoff frequency
def hp(cutoff):
    wc = 2 * np.pi * cutoff
    return AnCompFun(lambda f : 2 * np.pi * f/wc / (2 * np.pi * f/wc - 1j))

# Low Pass Filter with cutoff frequency
def lp(cutoff):
    wc = 2 * np.pi * cutoff
    return AnCompFun(lambda f: -1j / ((2 * np.pi * f/wc) - 1j))

# PI Transfer function with corner frequency and HF gain
def pi(corner, gain):
    return AnCompFun(lambda f: gain * (1 - 1.0j * corner / f))

# Harmonic Oscillator T.F. with resonant frequency and damping rate.
def ho(res, damp):
    wres = 2 * np.pi * res
    wdamp = 2 * np.pi * damp
    return AnCompFun(lambda f: 1 / (1 + 1j * 2 * np.pi * f * wdamp / wres**2 - ((2 * np.pi * f)/wres)**2))

# Lag Compensator with frequency and amplitude
def lag(ff,a):
    assert a <= 1 and a > 0, "Invalid 'a' value in lag filter"
    return AnCompFun(lambda f: a*(1 + 1j*f/ff)/(a + 1j * f/ff))

# Lead Compensator with frequency and amplitude
def lead(ff,a):
    assert a <= 1 and a > 0, "Invalid 'a' value in lead filter"
    return AnCompFun(lambda f: a*(1 + 1j*f/ff)/(a*1j*f/ff + 1))

# Flat gain amplification
def amp(a):
    return AnCompFun(lambda f: np.ones(np.size(f)) * a)

# Time delay
def delay(delta_t):
    return AnCompFun(lambda f: np.exp(-1j * 2 * np.pi * f * delta_t))

# LFGL filter defined by two resistances and two capacitances.
def lfgl(R1,R2,C1,C2):
    return AnCompFun(lambda f: (R2 * (-1j + C1*R1*2*np.pi*f))/(-1j*(R1+R2)+(C1+C2)*R1*R2*2*np.pi*f))

############
# Plotting #
############
def plot_trans(trans, lines=True, norm=False, unwrap=False):
    amp = trans['r']
    if norm:
        amp /= max(amp)

    phase = trans['phase']

    freq = trans['frequency']

    fig, axes = plt.subplots(2, 1, sharex = True, squeeze = True)
    plot_amp(axes[0], amp, freq, lines)
    plot_phase(axes[1], phase, freq, unwrap, lines)

    axes[0].set_ylim([min(amp)-np.power(10,np.round(np.log(min(amp)))),
                      max(amp)+np.power(10,np.floor(np.log(max(amp))-1))])

    return fig

def plot_amp(ax, amp, freq, lines=False):
    ax.plot(freq, amp)

    if lines:
        ax.plot(freq, np.ones_like(freq), linestyle='--', zorder=0)
        # Find Crossing Point
        cross = np.argmax(amp <= 1.0)
        # Plot 1 one pole slope crossing at cross point
        ax.plot(freq, freq[cross]/freq, linestyle='--', zorder=0)
        ax.plot(freq, freq[cross]**2/(freq**2), linestyle='--',zorder=0)

    # Set log axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")

def plot_phase(ax, phase, freq, unwrap=False, lines=False):
    if unwrap:
        phase = np.unwrap(phase)
    ax.plot(freq, phase/(2 * np.pi) * 360)
    if lines:
        ax.axhline(0,linestyle='--',color="gray")

    ax.set_xscale('log')
    ax.set_ylabel("Phase")
    ax.set_yticks([-180,-90,0,90,180])
    ax.set_xlabel("Frequency (Hz)")

def plot_funcs(funcs, freq=np.array([]), labels=[], unwrap=False, lines=False, **kwargs):
    total, axes = plt.subplots(2,1,sharex=True,squeeze=True,**kwargs)
    for index, func in enumerate(funcs):
        if isinstance(func, AnCompFun):
            if freq.size == 0:
                raise(ValueError("No frequencies provided for analytic function"))
            func = func.apply(freq).polar()
            plot_amp(axes[0], func['r'], freq, False)
            plot_phase(axes[1], func['phase'], freq, unwrap, False)
        else:
            func = func.polar()
            plot_amp(axes[0], func['r'], func['frequency'], False)
            plot_phase(axes[1], func['phase'], func['frequency'], unwrap, False)
        axes[0].legend(labels)
    
    phase_lims = axes[1].get_ylim()
    pos_nineties = np.ceil(phase_lims[1]/90)
    neg_nineties = np.floor(phase_lims[0]/90)
    if pos_nineties - neg_nineties > 10:
        axes[1].set_yticks(np.linspace(neg_nineties*90, pos_nineties*90+1,10))
    else:
        axes[1].set_yticks(np.arange(neg_nineties * 90, pos_nineties * 90+1 , 90))
    
    if lines:
        axes[0].axhline(1,linestyle="--", color="gray")
        axes[1].axhline(0,linestyle="--", color="gray")
    return total

############
# Analysis #
############
# These likely don't work.
def gain_margin(function):
    if isinstance(function, AnCompFun):
        z = opt.root_scalar(lambda f: np.angle(function.func(f)) + np.pi, bracket=[0,1E15], x0=1000.0)
        return np.abs(function.f(z))
    elif isinstance(function, CompFun):
        z = function.f[np.argmin(np.abs(np.angle(function.c)))]
        return np.abs(function.c[z])
    raise ValueError("Wrong Type")

def phase_margin(function):
    if isinstance(function, AnCompFun):
        z = opt.root_scalar(lambda f: np.abs(function.func(f)) - 1, bracket=[0.,1E15], x0=1000.0)
        return np.angle(function.f(z))
    elif isinstance(function, CompFun):
        z = function.f[np.argmin(np.abs(np.abs(function.c) - 1))]
        return np.angle(function.c[z])
    raise ValueError("Wrong Type")