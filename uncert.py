import numpy as np
import metrolopy as mp

def from_floats(xs, weights=None, unit=None):
    """
    Given a list of x values, returns a gummy with mean center
    and std error uncertainty. The calculation of the uncertainty is 
    based on Sect. 4.2 of the BIPM's 
    "Guide to the Expression of Uncertainty in Measurement"

    Parameters
    ----------
    xs : [number]
        The values to convert into a single mean with uncertainty.
    weights : [number], optional
        The weights to be considered for each x value when averaging
        and taking the standard deviation., by default None
    unit : str, optional
        The unit of the value and uncertainty to pass to gummy, by default None

    Returns
    -------
    gummy
        The gummy object containing the mean and uncertainty.
    """
    if weights is not None:
        if len(xs) != len(weights):
            raise RuntimeError("xs and weights must be of same length")

    mean = np.average(xs, weights=weights)
    # Calculate standard deviation, weighted if needed.
    # np.average divides sum of squared deviations by n. To properly compute
    # variance this should be (n-1)
    var = np.average((xs-mean)**2, weights=weights)*len(xs)/(len(xs)-1)
    # Standard error on mean is sqrt of (variance divided by n)
    stder = np.sqrt(var/len(xs))

    return mp.gummy(mean, stder, unit=unit)

def from_gummys(gummys):
    """
    To compute a properly weighted average from a list of gummy's
    simply take the center values and compute weights from uncertainties.
    Then use the above function.

    Parameters
    ----------
    gummys : [gummy]
        List of gummy values to compute a weighted mean with error from.

    Returns
    -------
    gummy
        weighted error with uncertainty given by standard error on mean.
    """
    if not len(gummys):
        raise ValueError("Empty List Provided")
    
    xs = [gummy.x for gummy in gummys]
    try:
        weights = [1/gummy.u for gummy in gummys]
    except ZeroDivisionError:
        print("Zero uncertainty encountered, weights set to None")
        weights = None
    return from_floats(xs, weights, unit=gummys[0].unit)

def from_fit(result):
    """
    Converts the results from a fit into a dict of gummys with names
    taken from the parameters.

    Parameters
    ----------
    result : lmfit.ModelResult
        The fit result from lmfit to extract the data from.
    """
    params = result.params
    return {name : mp.gummy(param.value,param.stderr) for name,param in params.items()}