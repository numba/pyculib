import numpy as np
from pyculib import warnings

promote = np.promote_types   # type promotion

def alias(a, b):
    """Check whether the arrays `a` and `b` alias."""

    if a is b:
        return True
    elif a.base is None and b.base is None:
        return False
    else:
        return a.base is b or a is b.base or a.base is b.base

def astype(x, dtype, var, stacklevel=3):
    """Return `x` or a copy of `x`, with its type converted to `dtype`.
    `var` is the name of `x` as seen by users of a public API, which may be
    used in a warning message. `stacklevel` corresponds to the number of frames
    to skip when reporting the warning."""

    # stacklevel=3 means the warning will be reported against the BLAS call,
    # not against this (astype()) function.
    # Make this a variable as sometimes the call is nested, so the number of
    # frames needs to be adjusted.
    if dtype != x.dtype:
        warnings.warn("%s (%s) is converted to %s"%(var, x.dtype, dtype),
                      warnings.PerformanceWarning, stacklevel=stacklevel)
    return x.astype(dtype, copy=False)


def colmajor(x, dtype, var):
    """Return `x` or a copy of `x`, with its dimension ordering converted to
    column-major, and its type converted to `dtype`.
    `var` is the name of `x` as seen by users of a public API, which may be
    used in a warning message."""

    if not x.flags['F_CONTIGUOUS']:
        warnings.warn("%s is converted to column-major layout"%(var),
                      warnings.PerformanceWarning, stacklevel=3)
        return np.asfortranarray(x, dtype=dtype)
    else:
        return astype(x, dtype, var, stacklevel=4)


