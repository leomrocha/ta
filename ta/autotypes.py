"""
This module will decide to use either 32 or 64 bits numba int and float types depending on the hardware architecture
"""

from archifind import archi
# from .archifind import archi

if archi == 64:
    from numba import int64, float64
    nb_int = int64
    nb_float = float64
else:  # assume 32 bits
    from numba import int32, float32
    nb_int = int32
    nb_float = float32

NUMBA_OPTS = {
              "cache": True,
              "fastmath": True,  # compromise speed for accuracy
              "nogil": True,
              # "nopython": True,  # use njit instead
              # "parallel": True,  # is not always a good idea due to context changes
              }
