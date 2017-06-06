from os.path import dirname, join
from . import (test_blas_low_level, test_blas, test_fft, test_rand,
               test_sorting, test_sparse)

test_cases = (
    test_blas_low_level.test_cases +
    test_blas.test_cases +
    test_fft.test_cases +
    test_rand.test_cases +
    test_sorting.test_cases +
    test_sparse.test_cases
)
