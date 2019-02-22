# -*- coding: utf-8 -*-
import math
import numpy as np

from numba import njit, vectorize, jitclass
from numba import int32, float32    # import the types

NUMBA_OPTS = {
              "cache": True,
              "fastmath": True,
              "nogil": True,
              # "nopython": True,  # using njit instead
              # "parallel": True,  # is not always a good idea due to context changes
              }

# import pandas as pd


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


# replace this funciton with a numpy based one
def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


# implementation from here:
# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm?noredirect=1&lq=1

# using the fastest implementation available there
# def numpy_ewma_vectorized_v2(data, window):
# @njit((float32[:], int32), fastmath=True, nogil=True)
@njit((float32[:], int32), **NUMBA_OPTS)
def ewma(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm


# @njit((float32[:], int32), fastmath=True, nogil=True)
@njit((float32[:], int32), **NUMBA_OPTS)
def _ewma(arr_in, window):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float32)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@njit((float32[:], int32), **NUMBA_OPTS)
def _ewma_infinite_hist(arr_in, window):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    assuming infinite history via the recursive form:

        (2) (i)  y[0] = x[0]; and
            (ii) y[t] = a*x[t] + (1-a)*y[t-1] for t>0.

    This method is less accurate that ``_ewma`` but
    much faster:

        In [1]: import numpy as np, bars
           ...: arr = np.random.random(100000)
           ...: %timeit bars._ewma(arr, 10)
           ...: %timeit bars._ewma_infinite_hist(arr, 10)
        3.74 ms ± 60.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        262 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=False).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float32)
    alpha = 2 / float(window + 1)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma


# exponential moving average inspired from here
# https://codereview.stackexchange.com/questions/70510/calculating-exponential-moving-average-iclass Indicators:
@njit((float32[:], int32), **NUMBA_OPTS)
def ema2(values, window):
    """
    Numpy implementation of EMA
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

#
# # although I have to dig more into it like here:
# # https://stackoverflow.com/questions/52783479/how-does-pandas-compute-exponential-moving-averages-under-the-hood
# @njit((float32[:], int32), **NUMBA_OPTS)
# def ewm(arr, alpha):
#     """
#     Calculate the EMA of an array arr
#     :param arr: numpy array of floats
#     :param alpha: float between 0 and 1
#     :return: numpy array of floats
#     """
#     # initialise ewm_arr
#     ewm_arr = np.zeros_like(arr)
#     ewm_arr[0] = arr[0]
#     for t in range(1, arr.shape[0]):
#         ewm_arr[t] = alpha*arr[t] + (1 - alpha)*ewm_arr[t-1]
#
#     return ewm_arr
