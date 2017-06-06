from __future__ import print_function, absolute_import, division

import unittest
import numpy as np
import scipy.linalg
from pyculib import warnings, config
from numba.testing.ddt import ddt, unpack, data
import time

def create_array(dtype, shape, slices=None, empty=False):
    """Create a test array of the given dtype and shape.
    if slices is given, the returned array aliases a bigger parent array
    using the specified start and step values. (The stop member is expected to
    be appropriate to yield the given length.)"""

    from numpy.random import normal, seed
    seed(1234)

    def total_size(s):
        # this function doesn't support slices whose members are 'None'
        return s.start + (s.stop - s.start)*np.abs(s.step)

    if not slices:
        a = np.empty(dtype=dtype, shape=shape)
    else:
        if type(shape) is not tuple: # 1D
            pshape = total_size(slices)
        else:
            pshape = tuple([total_size(s) for s in slices])
        parent = np.empty(dtype=dtype, shape=pshape)
        a = parent[slices]

    if not empty:
        mult = np.array(1, dtype=dtype)
        a[:] = normal(0.,1.,shape).astype(dtype) * mult
    return a


class TestBLAS:
    """Create test cases by deriving from this (as well as unittest.TestCase.
    In the derived class, set the 'blas' attribute to the BLAS module that is
    to be tested."""

    blas = None

    def _test_dot(self, dtype, shape, slice, rtol=1e-07):

        x = create_array(dtype, shape, slice)
        y = create_array(dtype, shape, slice)
        res = self.blas.dot(x, y)
        ref = np.dot(x, y)
        np.testing.assert_allclose(res, ref, rtol=rtol)

    def _test_gemv(self, dtype, op, shape, slices, rtol=1e-07):

        sp_gemv = scipy.linalg.get_blas_funcs('gemv', dtype=dtype)
        # f2py convention...
        sp_trans = {'N':0, 'T':1, 'C':2}

        alpha = 2.
        A = create_array(dtype, shape, slices, empty=True)
        A[:] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        x = np.arange(op == 'N' and shape[1] or shape[0], dtype=dtype)
        beta = 2.
        y = np.arange(op == 'N' and shape[0] or shape[1], dtype=dtype)
        res = self.blas.gemv(op, alpha, A, x, beta, y)
        y = np.arange(op == 'N' and shape[0] or shape[1], dtype=dtype)
        ref = sp_gemv(alpha, A, x, beta, y, trans=sp_trans[op])
        np.testing.assert_allclose(res, ref, rtol=rtol)

    def _test_axpy(self, dtype, size, slice, rtol=1e-07):

        sp_axpy = scipy.linalg.get_blas_funcs('axpy', dtype=dtype)
        alpha = 2.
        x = create_array(dtype, size, slice)
        y = create_array(dtype, size, slice)
        yr = np.copy(y)
        res = self.blas.axpy(alpha, x, y)
        ref = sp_axpy(x, yr, size, alpha)
        np.testing.assert_allclose(res, ref, rtol=rtol)

    def _test_gemm(self, dtype, opa, opb, ashape, aslices, bshape, bslices, rtol=1e-07):

        M = opa == 'N' and ashape[0] or ashape[1]
        N = opb == 'N' and bshape[1] or bshape[0]
        cshape = (M, N)
        sp_gemm = scipy.linalg.get_blas_funcs('gemm', dtype=dtype)
        # f2py convention...
        sp_trans = {'N':0, 'T':1, 'C':2}

        alpha = 2.
        A = create_array(dtype, ashape, aslices, empty=True)
        A[:] = np.arange(np.prod(ashape), dtype=dtype).reshape(ashape)
        B = create_array(dtype, bshape, bslices, empty=True)
        B[:] = np.arange(np.prod(bshape), dtype=dtype).reshape(bshape)
        beta = 5.
        C = create_array(dtype, cshape, empty=True)
        C[:] = np.arange(np.prod(cshape), dtype=dtype).reshape(cshape)
        res = self.blas.gemm(opa, opb, alpha, A, B, beta, C)
        # C may have been overwritten in the previous operation.
        C = np.arange(np.prod(cshape), dtype=dtype).reshape(cshape)
        ref = sp_gemm(alpha, A, B, beta, C, trans_a=sp_trans[opa], trans_b=sp_trans[opb])
        np.testing.assert_allclose(res, ref, rtol=rtol)

    def test_dot_invalid(self):

        x = np.arange(1024, dtype=np.float32)
        y = np.arange(1024, dtype=np.float32)
        # First make sure the original works...
        res = self.blas.dot(x, y)
        ref = np.dot(x, y)
        np.testing.assert_allclose(res, ref, rtol=1e6)
        # then check for various types of invalid input
        with self.assertRaises(TypeError): # invalid type
          self.blas.dot(np.arange(10), np.arange(10))
        with self.assertRaises(ValueError):
            self.blas.dot(x, y.reshape(64, 16)) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.dot(x, y[4:]) # invalid size

    def test_axpy_invalid(self):

        # First make sure the original works...
        sp_axpy = scipy.linalg.get_blas_funcs('axpy', dtype=np.float32)
        alpha = np.float32(2.)
        x = np.arange(64, dtype=np.float32)
        y = np.arange(64, dtype=np.float32)
        res = self.blas.axpy(alpha, x, y)
        y = np.arange(64, dtype=np.float32)
        ref = sp_axpy(x, y, 64, alpha)
        np.testing.assert_allclose(res, ref)
        # then check for various types of invalid input
        with self.assertRaises(TypeError): # invalid type
          self.blas.axpy(7, np.arange(64), y)
        with self.assertRaises(TypeError): # invalid type
          self.blas.axpy(7, x, np.arange(64))
        with self.assertRaises(ValueError):
            self.blas.axpy([1], x, y) # invalid scalar
        with self.assertRaises(ValueError):
            self.blas.axpy(alpha, x, y.reshape(8, 8)) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.axpy(alpha, x, y[4:]) # invalid size

    def test_gemv_invalid(self):

        # First make sure the original works...
        sp_gemv = scipy.linalg.get_blas_funcs('gemv', dtype=np.float32)
        alpha = 2.
        A = np.arange(64, dtype=np.float32).reshape(8,8)
        x = np.arange(8, dtype=np.float32)
        beta = 2.
        y = np.arange(8, dtype=np.float32)
        res = self.blas.gemv('N', alpha, A, x, beta, y)
        y = np.arange(8, dtype=np.float32)
        ref = sp_gemv(alpha, A, x, beta, y)
        np.testing.assert_allclose(res, ref)
        # then check for various types of invalid input
        i8x8 = np.arange(64).reshape(8,8)
        i8 = np.arange(8)
        with self.assertRaises(TypeError):
            self.blas.gemv('N', alpha, i8x8, x, beta, y) # invalid type
        with self.assertRaises(TypeError):
            self.blas.gemv('N', alpha, A, i8, beta, y) # invalid type
        with self.assertRaises(TypeError):
            self.blas.gemv('N', alpha, A, x, beta, i8) # invalid type
        with self.assertRaises(ValueError):
            self.blas.gemv('X', alpha, A, x, beta, y) # invalid op
        with self.assertRaises(ValueError):
            self.blas.gemv( 'N', [1], A, x, beta, y) # invalid scalar
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A[0], x) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A, x.reshape(2, 4), beta, y) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A, x, beta, y.reshape(2, 4)) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A.reshape(64), x, beta, y) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A[1:,:], x, beta, y) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A, x[1:], beta, y) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemv('N', alpha, A, x, beta, y[1:]) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemv('T', alpha, A, x[1:], beta, y) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemv('T', alpha, A, x, beta, y[1:]) # invalid size


    def test_gemm_invalid(self):

        # First make sure the original works...
        sp_gemm = scipy.linalg.get_blas_funcs('gemm', dtype=np.float32)

        alpha = 2.
        A = np.arange(64, dtype=np.float32).reshape(8, 8)
        B = np.arange(64, dtype=np.float32).reshape(8, 8)
        beta = 5.
        C = np.arange(64, dtype=np.float32).reshape(8, 8)
        res = self.blas.gemm('N', 'N', alpha, A, B, beta, C)
        # C may have been overwritten in the previous operation.
        C = np.arange(64, dtype=np.float32).reshape(8, 8)
        ref = sp_gemm(alpha, A, B, beta, C)
        np.testing.assert_allclose(res, ref)
        # then check for various types of invalid input
        i8x8 = np.arange(64).reshape(8,8)
        with self.assertRaises(TypeError):
            self.blas.gemm('N', 'N', alpha, i8x8, B, beta, C) # invalid type
        with self.assertRaises(TypeError):
            self.blas.gemm('N', 'N', alpha, A, i8x8, beta, C) # invalid type
        with self.assertRaises(TypeError):
            self.blas.gemm('N', 'N', alpha, A, B, beta, i8x8) # invalid type
        with self.assertRaises(ValueError):
            self.blas.gemm('X', 'N', alpha, A, B, beta, C) # invalid op
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'X', alpha, A, B, beta, C) # invalid op
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', [1], A, B, beta, C) # invalid scalar
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B, [1], C) # invalid scalar
        with self.assertRaises(ValueError):
            self.blas.gemm('C', 'N', alpha, A[0], B) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B[0]) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A.reshape(64), B, beta, C) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B.reshape(64), beta, C) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B, beta, C.reshape(64)) # invalid dim
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A[1:,:], B, beta, C) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B[1:,:], beta, C) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B, beta, C[1:,:]) # invalid size
        with self.assertRaises(ValueError):
            self.blas.gemm('N', 'N', alpha, A, B, beta, C[:,1:]) # invalid size

    def test_gemv_default(self):

        # Check that default argument and aliasing rules work as expected
        alpha = 2.
        A = np.arange(64, dtype=np.float64).reshape(8,8)
        x = np.arange(8, dtype=np.float64)
        beta = 0.
        yres = self.blas.gemv('N', alpha, A, x)
        y = np.asfortranarray(np.arange(8, dtype=np.float64))
        res = self.blas.gemv('N', alpha, A, x, beta, y)
        # Make sure the result is the same even with no default y...
        np.testing.assert_allclose(yres, res)
        # ...and res indeed aliases y
        self.assertIs(res, y)
        # Make sure this also works for non-contiguous y
        p = np.arange(16, dtype=np.float64) / 2
        y = p[::2]
        res = self.blas.gemv('N', alpha, A, x, beta, y)
        np.testing.assert_allclose(yres, y)
        np.testing.assert_allclose(res, y)

    def test_gemm_default(self):

        # Check that default argument and aliasing rules work as expected
        alpha = 2.
        A = np.arange(64, dtype=np.float64).reshape(8,8)
        B = np.arange(64, dtype=np.float64).reshape(8,8)
        beta = 0.
        Cres = self.blas.gemm('N', 'N', alpha, A, B)
        C = np.arange(64, dtype=np.float64).reshape(8, 8, order='F')
        res = self.blas.gemm('N', 'N', alpha, A, B, beta, C)
        # Make sure the result is the same even with no default C...
        np.testing.assert_allclose(Cres, res)
        # ...and res indeed aliases C
        self.assertIs(res, C)
        # Make sure this also works for non-contiguous C
        p = np.arange(256, dtype=np.float64).reshape(16, 16) / 2
        C = p[::2,::2]
        res = self.blas.gemm('N', 'N', alpha, A, B, beta, C)
        np.testing.assert_allclose(Cres, C)
        np.testing.assert_allclose(res, C)

    def test_dot_type_promotion(self):

        #Make sure the result has the appropriate type for mixed input types.
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float32)
        self.assertIs(type(self.blas.dot(x, y)), np.float64)
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.complex64)
        self.assertIs(type(self.blas.dot(x, y)), np.complex128)
        
    @unittest.skipIf(not config.WARNINGS, "warnings are disabled")
    def test_dot_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            x = np.arange(4, dtype=np.float64)
            y = np.arange(4, dtype=np.float32)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.dot(x, y)
    
    def test_axpy_type_promotion(self):

        #Make sure the result has the appropriate type for mixed input types.
        alpha = 2.
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float32)
        self.assertIs(self.blas.axpy(alpha, x, y).dtype.type, np.float64)
        x = x.astype(np.complex64)
        self.assertIs(self.blas.axpy(alpha, x, y).dtype.type, np.complex128)
        alpha = 2.+1j
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float32)
        self.assertIs(self.blas.axpy(alpha, x, y).dtype.type, np.complex128)

    @unittest.skipIf(not config.WARNINGS, "warnings are disabled")
    def test_axpy_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            alpha = 1.
            x = np.arange(4, dtype=np.float64)
            y = np.arange(4, dtype=np.float32)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.axpy(alpha, x, y) # type promotion
            y = y.astype(np.float64)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.axpy(alpha, x[::2], y[::2]) # non-unit-stride

    def test_gemv_type_promotion(self):

        #Make sure the result has the appropriate type for mixed input types.
        alpha = 2.
        A = np.arange(16, dtype=np.float64).reshape(4,4)
        x = np.arange(4, dtype=np.float64)
        beta = 0.
        self.assertIs(self.blas.gemv('N', alpha, A, x).dtype.type, np.float64)
        x = x.astype(np.complex64)
        self.assertIs(self.blas.gemv('N', alpha, A, x).dtype.type, np.complex128)
        y = np.asfortranarray(np.arange(4, dtype=np.float64))
        self.assertIs(self.blas.gemv('N', alpha, A, x, beta, y).dtype.type, np.complex128)

    @unittest.skipIf(not config.WARNINGS, "warnings are disabled")
    def test_gemv_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            alpha = 1.
            A = np.arange(16, dtype=np.float64).reshape(4,4)
            x = np.arange(4, dtype=np.float32)
            beta = 0.
            y = np.arange(4, dtype=np.float32)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemv('N', alpha, A, x) # type promotion
            x = x.astype(np.float64)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemv('N', alpha, A, x, beta, y) # type promotion
            y = y.astype(np.float64)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemv('N', alpha, A[::2,::2], x[::2], beta, y[::2]) # non-unit-stride

    def test_gemm_type_promotion(self):

        #Make sure the result has the appropriate type for mixed input types.
        alpha = 2.
        A = np.arange(16, dtype=np.float64).reshape(4,4)
        B = np.arange(16, dtype=np.float64).reshape(4,4)
        beta = 0.
        self.assertIs(self.blas.gemm('N', 'N', alpha, A, B).dtype.type, np.float64)
        A = A.astype(np.complex64)
        self.assertIs(self.blas.gemm('N', 'N', alpha, A, B).dtype.type, np.complex128)
        C = np.asfortranarray(np.arange(16, dtype=np.float64).reshape(4,4))
        self.assertIs(self.blas.gemm('N', 'N', alpha, A, B, beta, C).dtype.type, np.complex128)
        C = C.astype(np.complex128)
        self.assertIs(self.blas.gemm('N', 'N', alpha, A, B, beta, C).dtype.type, np.complex128)

    @unittest.skipIf(not config.WARNINGS, "warnings are disabled")
    def test_gemm_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            alpha = 1.
            A = np.arange(16, dtype=np.float32).reshape(4,4)
            B = np.arange(16, dtype=np.float64).reshape(4,4)
            beta = 0.
            C = np.arange(16, dtype=np.complex64).reshape(4,4)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemm('N', 'N', alpha, A, B) # type promotion
            A = A.astype(np.float64)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemm('N', 'N', alpha, A, B, beta, C) # type promotion
            C = np.arange(16, dtype=np.float64).reshape(4,4)
            with self.assertRaises(warnings.PerformanceWarning):
                self.blas.gemm('N', 'N', alpha, A[::2,::2], B[::2,::2], beta, C[::2,::2]) # non-unit-stride

        

if __name__ == '__main__':
    unittest.main()
