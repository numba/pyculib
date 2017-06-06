from __future__ import absolute_import, print_function
from .api import Blas, validate_argument_dtype
from pyculib.nputil import promote, alias, astype, colmajor
import numpy as np
from numbers import Number


_blas = None

def _normalize_op(op):

    if op == 'n': return 'N'
    elif op == 't': return 'T'
    elif op == 'c': return 'C'
    return op


def dot(x, y, stream=None):
    """Compute and return the vector dot product of x and y."""
    global _blas

    validate_argument_dtype(x, 'x')
    validate_argument_dtype(y, 'y')
    if not _blas: _blas = Blas()
    _blas.stream = stream
    dtype = promote(x.dtype, y.dtype)
    # FIXME: the col-major constraint actually only applies to host arrays.
    #        If x and y are already device arrays they can be strided.
    return _blas.dot(colmajor(x, dtype, 'x'), colmajor(y, dtype, 'y'))

def axpy(alpha, x, y, stream=None):
    """y <- alpha*x + y """

    global _blas

    if not isinstance(alpha, Number): raise ValueError('alpha is not a numeric type')
    validate_argument_dtype(x, 'x')
    validate_argument_dtype(y, 'y')
    if not _blas: _blas = Blas()
    _blas.stream = stream
    dtype = promote(promote(type(alpha), x.dtype), y.dtype)
    yf = colmajor(y, dtype, 'y')
    _blas.axpy(dtype.type(alpha), x.astype(dtype), yf)
    if y.dtype == yf.dtype and not alias(y, yf):
        y[:] = yf
        return y
    else:
        return yf

def gemv(trans, alpha, A, x, beta=0, y=None, stream=None):
    """Generalized matrix-vector multiplication:

    y <- alpha*trans(A)*x + beta*y

    'beta' and 'y' are optional on input. Return 'y'."""

    global _blas

    if not isinstance(alpha, Number): raise ValueError('alpha is not a numeric type')
    validate_argument_dtype(A, 'A')
    validate_argument_dtype(x, 'x')
    if not isinstance(beta, Number): raise ValueError('beta is not a numeric type')
    if A.ndim != 2: raise ValueError('A is not a two-dimensional array')
    if x.ndim != 1: raise ValueError('x is not a one-dimensional array')
    if not _blas: _blas = Blas()
    _blas.stream = stream
    m, n = A.shape
    trans = _normalize_op(trans)
    if trans not in ('N', 'T', 'C'): raise ValueError('trans has invalid value')
    dtype = promote(promote(type(alpha), A.dtype),
                    promote(x.dtype, type(beta)))
    if y is None:
        y = np.empty(trans == 'N' and n or m, dtype=dtype)
        yf = y
    else:
        validate_argument_dtype(y, 'y')
        if y.ndim != 1: raise ValueError('y is not a one-dimensional array')
        dtype = promote(dtype, y.dtype)
        yf = colmajor(y, dtype, 'y')
    if trans == 'N':
        if A.shape[1] != x.shape[0]:
            raise ValueError('arrays A and x have incompatible shapes')
        if A.shape[0] != y.shape[0]:
            raise ValueError('arrays A and y have incompatible shapes')
    else:
        if A.shape[0] != x.shape[0]:
            raise ValueError('arrays A and x have incompatible shapes')
        if A.shape[1] != y.shape[0]:
            raise ValueError('arrays A and y have incompatible shapes')
    _blas.gemv(trans, m, n, dtype.type(alpha), colmajor(A, dtype, 'A'),
               x.astype(dtype), dtype.type(beta), yf)
    if y.dtype == yf.dtype and not alias(y, yf):
        y[:] = yf
        return y
    else:
        return yf

def gemm(transa, transb, alpha, A, B, beta=0, C=None, stream=None):
    """Generalized matrix-matrix multiplication:

    C <- alpha*transa(A)*transb(B) + beta*C

    'beta' and 'C' are optional on input. Return 'C'."""

    global _blas

    if not isinstance(alpha, Number): raise ValueError('alpha is not a numeric type')
    validate_argument_dtype(A, 'A')
    validate_argument_dtype(B, 'B')
    if not isinstance(beta, Number): raise ValueError('beta is not a numeric type')
    if A.ndim != 2: raise ValueError('A is not a two-dimensional array')
    if B.ndim != 2: raise ValueError('B is not a two-dimensional array')
    if not _blas: _blas = Blas()
    _blas.stream = stream
    transa = _normalize_op(transa)
    if transa not in ('N', 'T', 'C'): raise ValueError('transa has invalid value')
    transb = _normalize_op(transb)
    if transb not in ('N', 'T', 'C'): raise ValueError('transb has invalid value')
    dtype = promote(promote(type(alpha), A.dtype),
                    promote(B.dtype, type(beta)))
    M = transa == 'N' and A.shape[0] or A.shape[1]
    N = transb == 'N' and B.shape[1] or B.shape[0]
    K = transa == 'N' and A.shape[1] or A.shape[0]
    if C is None:
        C = np.empty(shape=(M, N), order='F', dtype=dtype)
        Cf = C
    else:
        validate_argument_dtype(C, 'C')
        if C.ndim != 2: raise ValueError('C is not a two-dimensional array')
        if C.shape[0] != M:
            raise ValueError('arrays A and C have incompatible shapes')
        if C.shape[1] != N:
            raise ValueError('arrays B and C have incompatible shapes')
        dtype = promote(dtype, C.dtype)
        Cf = colmajor(C, dtype, 'C')

    if transb == 'N':
        if B.shape[0] != K: raise ValueError('arrays A and B have incompatible shapes')
    else:
        if B.shape[1] != K: raise ValueError('arrays A and B have incompatible shapes')

    _blas.gemm(transa, transb, M, N, K, dtype.type(alpha),
               colmajor(A, dtype, 'A'), colmajor(B, dtype, 'B'),
               dtype.type(beta), Cf)
    if C.dtype == Cf.dtype and not alias(C, Cf):
        C[:] = Cf
        return C
    else:
        return Cf
