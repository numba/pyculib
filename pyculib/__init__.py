from __future__ import absolute_import
import numba
import re
import unittest
import sys
import platform
from pyculib import config
from pyculib import warnings

NUMBA_VERSION_REQ = (0, 33, 0)

def check_numba_version():
    m = re.match(r"(\d+)\.(\d+)\.(\d+).*", numba.__version__)
    if m is None or tuple(map(int, m.groups())) < NUMBA_VERSION_REQ:
        import warnings
        warnings.showwarning(
            "Numba version too old; expecting %d.%d.%d" % NUMBA_VERSION_REQ,
            ImportWarning, __name__, 1)

check_numba_version()

def load_tests(loader, tests, pattern):
    from .tests import test_cases

    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite

def cuda_compatible():
    if sys.platform.startswith('darwin'):
        ver = platform.mac_ver()[0]
        # version string can contain two or three components
        major, minor = ver.split('.', 1)
        if '.' in minor:
            minor, micro = minor.split('.', 1)
        if (int(major), int(minor)) < (10, 9):
            return False

    is_64bits = sys.maxsize > 2**32
    if not is_64bits:
        return False

    return True

if cuda_compatible():
    from numba import cuda
    from . import blas, sparse, fft, rand, sorting

def test():
    success = True
    if cuda_compatible() and cuda.is_available():
        print('CUDA Library tests'.center(80, '~'), '\n')
        print('cuBLAS'.center(80, '-'))
        if not blas.test().wasSuccessful():
            success = False
        print('cuSPARSE'.center(80, '-'))
        if not sparse.test().wasSuccessful():
            success = False
        print('cuFFT'.center(80, '-'))
        if not fft.test().wasSuccessful():
            success = False
        print('cuRAND'.center(80, '-'))
        if not rand.test().wasSuccessful():
            success = False
        print('Sorting'.center(80, '-'))
        if not sorting.test().wasSuccessful():
            success = False
    else:
        print('CUDA unavailable - skipped CUDA tests')

    return success

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
