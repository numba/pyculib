from __future__ import absolute_import
from pyculib import config
import warnings # so we can use warnings.warn
from warnings import *

class PerformanceWarning(Warning):
    """
    Warning category for when an operation might not be
    as fast as expected.
    """


# Define a simple no-op for the (default) case
# where performance warnings are disabled.

def no_warn(*args, **kwds): pass

if config.WARNINGS:
    warn = warnings.warn

else:
    warn = no_warn
