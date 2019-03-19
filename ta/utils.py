# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from numba import njit, vectorize, jitclass
# from .autotypes import *
from autotypes import *


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


# @njit((nb_float[:], nb_int), **NUMBA_OPTS)
def ewma(data, window):
    r"""
    Exponentialy weighted moving average specified by a decay ``window``

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param data: np.ndarray A single dimensional numpy array
    :param window: the decay window (span)
    :return: np.ndarray EWMA vector of shape data.shape

    >>> import pandas as pd
    >>> a = np.arange(15, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=False).mean()
    >>> np.array_equal(ewma(a, 10), exp.values.ravel())
    True
    >>>
    """

    alpha = 2. / (window + 1.)
    alpha_rev = 1-alpha
    n = data.shape[0]
    assert (n > 1)
    out = np.empty(n, dtype=np.float)
    out[0] = data[0]

    for i in range(1, n):
        out[i] = data[i] * alpha + out[i-1] * alpha_rev
    return out
