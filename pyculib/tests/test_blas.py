from __future__ import print_function, absolute_import, division

import unittest
import numpy as np
from numba.testing.ddt import ddt, unpack, data
from pyculib.tests.base import CUDATestCase
from pyculib import blas as cublas
from pyculib.tests import blas

@ddt
class TestCUDABLAS(blas.TestBLAS, CUDATestCase):

    blas = cublas
    
    @data((np.float32, 1024, slice(0, 1024, 1)),
          (np.float64, 1024, slice(0, 1024, 1)),
          (np.complex64, 1024, slice(0, 1024, 1)),
          (np.complex128, 1024, slice(0, 1024, 1)),
          (np.complex128, 1024, slice(1, 1025, 1)),
          (np.float32, 1024, slice(0, 2048, 2)),
          (np.float64, 1024, slice(0, 2048, 2)),
          (np.complex64, 1024, slice(0, 2048, 2)),
          (np.complex128, 1024, slice(0, 2048, 2)))
    @unpack
    def test_dot(self, dtype, shape, slice):

        eps = np.finfo(dtype).eps
        self._test_dot(dtype, shape, slice, rtol=eps*10)

    @data((np.float32, 'N', (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.float64, 'N', (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.float64, 'T', (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex64, 'N', (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex128, 'N', (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex128, 'N', (64, 67), (slice(1, 65, 1), slice(2, 69, 1))),
          (np.complex128, 'T', (67, 64), (slice(2, 69, 1), slice(1, 65, 1))),
          (np.complex128, 'C', (67, 64), (slice(2, 69, 1), slice(1, 65, 1))),
          (np.float32, 'N', (64, 67), (slice(0, 128, 2), slice(0, 134, 2))),
          (np.float64, 'N', (64, 67), (slice(0, 128, 2), slice(0, 134, 2))),
          (np.complex64, 'N', (64, 67), (slice(0, 128, 2), slice(0, 134, 2))),
          (np.complex128, 'N', (64, 67), (slice(0, 128, 2), slice(0, 134, 2)))
    )
    @unpack
    def test_gemv(self, dtype, op, shape, slices):

        self._test_gemv(dtype, op, shape, slices)

    @data((np.float32, 64, slice(0, 64, 1)),
          (np.float64, 64, slice(0, 64, 1)),
          (np.complex64, 64, slice(0, 64, 1)),
          (np.complex128, 64, slice(0, 64, 1)),
          (np.complex128, 64, slice(1, 65, 1)),
          (np.float32, 64, slice(0, 128, 2)),
          (np.float64, 64, slice(0, 128, 2)),
          (np.complex64, 64, slice(0, 128, 2)),
          (np.complex128, 64, slice(0, 128, 2)))
    @unpack
    def test_axpy(self, dtype, size, slice):

        self._test_axpy(dtype, size, slice)

    @data((np.float32, 'N', 'N',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (67, 63), (slice(0, 67, 1), slice(0, 63, 1))),
          (np.float64, 'N', 'N',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.float64, 'T', 'N',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.float64, 'N', 'T',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.float64, 'T', 'T',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex64, 'N', 'N',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.complex64, 'T', 'N',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.complex64, 'N', 'T',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex64, 'T', 'T',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex64, 'C', 'N',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.complex64, 'N', 'C',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex64, 'C', 'C',
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1)),
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1))),
          (np.complex128, 'N', 'N',
           (64, 67), (slice(0, 64, 1), slice(0, 67, 1)),
           (67, 64), (slice(0, 67, 1), slice(0, 64, 1))),
          (np.complex128, 'N', 'N',
           (64, 67), (slice(1, 65, 1), slice(2, 69, 1)),
           (67, 64), (slice(1, 68, 1), slice(2, 66, 1))),
          (np.float32, 'N', 'N',
           (64, 65), (slice(0, 128, 2), slice(0, 130, 2)),
           (65, 63), (slice(0, 130, 2), slice(0, 126, 2))),
          (np.float64, 'N', 'N',
           (64, 65), (slice(0, 128, 2), slice(0, 130, 2)),
           (65, 63), (slice(0, 130, 2), slice(0, 126, 2))),
          (np.complex64, 'N', 'N',
           (64, 65), (slice(0, 128, 2), slice(0, 130, 2)),
           (65, 63), (slice(0, 130, 2), slice(0, 126, 2))),
          (np.complex128, 'N', 'N',
           (64, 65), (slice(0, 128, 2), slice(0, 130, 2)),
           (65, 63), (slice(0, 130, 2), slice(0, 126, 2))))
    @unpack
    def test_gemm(self, dtype, opa, opb, ashape, aslices, bshape, bslices):

        eps = np.finfo(dtype).eps
        self._test_gemm(dtype, opa, opb, ashape, aslices, bshape, bslices, rtol=eps*10)


test_cases = (TestCUDABLAS,)

if __name__ == '__main__':
    unittest.main()
