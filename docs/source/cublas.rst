cuBLAS
======

Provides basic linear algebra building blocks. See `NVIDIA cuBLAS
<http://docs.nvidia.com/cuda/cublas/index.html>`_.

The cuBLAS binding provides an interface that accepts NumPy arrays and Numba's
CUDA device arrays. The binding automatically transfers NumPy array arguments to
the device as required. This automatic transfer may generate some unnecessary
transfers, so optimal performance is likely to be obtained by the manual
transfer for NumPy arrays into device arrays and using the cuBLAS to manipulate
device arrays where possible.

No special naming convention is used to identify the data
type, unlike in the BLAS C and Fortran APIs. Arguments for array storage
information which are part of the cuBLAS C API are also not necessary since
NumPy arrays and device arrays contain this information.

All functions are accessed through the :class:`pyculib.blas.Blas` class:

.. autoclass:: pyculib.blas.Blas

BLAS Level 1
------------

.. py:method:: pyculib.blas.Blas.nrm2(x)

    Computes the L2 norm for array `x`. Same as `numpy.linalg.norm(x)`.

    :param x: input vector
    :type x: python.array
    :returns: resulting norm.

.. py:method:: pyculib.blas.Blas.dot(x, y)

    Compute the dot product of array `x` and array `y`.  Same as `np.dot(x, y)`.

    :param x: vector
    :type x: python.array
    :param y: vector
    :type y: python.array
    :returns: dot product of `x` and `y`

.. py:method:: pyculib.blas.Blas.dotc(x, y)

    Uses the conjugate of the element of the vectors to compute the dot product
    of array `x` and array `y` for complex dtype only.  Same as `np.vdot(x, y)`.

    :param x: vector
    :type x: python.array
    :param y: vector
    :type y: python.array
    :returns: dot product of `x` and `y`


.. py:method:: pyculib.blas.Blas.scal(alpha, x)

    Scale `x` inplace by alpha.  Same as `x = alpha * x`

    :param alpha: scalar
    :param x: vector
    :type x: python.array

.. py:method:: pyculib.blas.Blas.axpy(alpha, x)

    Compute `y = alpha * x + y` inplace.

    :param alpha: scalar
    :param x: vector
    :type x: python.array


.. py:method:: pyculib.blas.Blas.amax(x)


    Find the index of the first largest element in array `x`.
    Same as `np.argmax(x)`

    :param x: vector
    :type x: python.array
    :returns: index (start from 0).


.. py:method:: pyculib.blas.Blas.amin(x)

    Find the index of the first largest element in array `x`.
    Same as `np.argmin(x)`

    :param x: vector
    :type x: python.array
    :returns: index (start from 0).


.. py:method:: pyculib.blas.Blas.asum(x)

    Compute the sum of all element in array `x`.

    :param x: vector
    :type x: python.array
    :returns: `x.sum()`

.. py:method:: pyculib.blas.Blas.rot(x, y, c, s)

    Apply the Givens rotation matrix specified by the cosine element `c` and the
    sine element `s` inplace on vector element `x` and `y`.

    Same as `x, y = c * x + s * y, -s * x + c * y`

    :param x: vector
    :type x: python.array
    :param y: vector
    :type y: python.array


.. py:method:: pyculib.blas.Blas.rotg(a, b)

    Constructs the Givens rotation matrix with the column vector (a, b).

    :param a: first element of the column vector
    :param b: second element of the column vector
    :returns: a tuple (r, z, c, s)

        r -- `r = a**2 + b**2`

        z -- Use to reconstruct `c` and `s`.
             Refer to cuBLAS documentation for detail.

        c -- The consine element.

        s -- The sine element.


.. py:method:: pyculib.blas.Blas.rotm(x, y, param)

    Applies the modified Givens transformation inplace.

    Same as::

        param = flag, h11, h21, h12, h22
        x[i] = h11 * x[i] + h12 * y[i]
        y[i] = h21 * x[i] + h22 * y[i]

    Refer to the cuBLAS documentation for the use of `flag`.

    :param x: vector
    :type x: python.array
    :param y: vector
    :type y: python.array


.. py:method:: pyculib.blas.Blas.rotmg(d1, d2, x1, y1)

    Constructs the modified Givens transformation `H` that zeros out the second
    entry of a column vector `(d1 * x1, d2 * y1)`.

    :param d1: scaling factor for the x-coordinate of the input vector
    :param d2: scaling factor for the y-coordinate of the input vector
    :param x1: x-coordinate of the input vector
    :param y1: y-coordinate of the input vector

    :returns: A 1D array that is usable in `rotm`.
              The first element is the flag for `rotm`.
              The rest of the elements corresponds to the `h11, h21, h12, h22`
              elements of `H`.

BLAS Level 2
-------------

All level 2 routines follow the following naming convention for all arguments:

* A, B, C, AP -- (2D array) Matrix argument.
                 `AP` implies packed storage for banded matrix.
* x, y, z -- (1D arrays)  Vector argument.
* alpha, beta -- (scalar) Can be floats or complex numbers depending.
* m -- (scalar)  Number of rows of matrix `A`.
* n -- (scalar)  Number of columns of matrix `A`.  If `m` is not needed,
                 `n` also means the number of rows of the matrix `A`; thus,
                 implying a square matrix.
* trans, transa, transb -- (string)
                Select the operation `op` to apply to a matrix:

                - 'N': `op(X) = X`, the identity operation;
                - 'T': `op(X) = X**T`, the transpose;
                - 'C': `op(X) = X**H`, the conjugate transpose.

                `trans` only applies to the only matrix argument.
                `transa` and `transb` apply to matrix `A` and matrix `B`,
                respectively.
* uplo -- (string) Can be 'U' for filling the upper trianglar matrix; or 'L' for
          filling the lower trianglar matrix.
* diag -- (boolean)  Whether the matrix diagonal has unit elements.
* mode -- (string) 'L' means the matrix is on the left side in the equation.
                   'R' means the matrix is on the right side in the equation.

.. note:: The last array argument is always overwritten with the result.

.. py:method:: pyculib.blas.Blas.gbmv(trans, m, n, kl, ku, alpha, A, x, beta, y)

    banded matrix-vector multiplication `y = alpha * op(A) * x + beta * y` where
    `A` has `kl` sub-diagonals and `ku` super-diagonals.

.. py:method:: pyculib.blas.Blas.gemv(trans, m, n, alpha, A, x, beta, y)

    matrix-vector multiplication `y = alpha * op(A) * x + beta * y`

.. py:method:: pyculib.blas.Blas.trmv(uplo, trans, diag, n, A, x)

    triangular matrix-vector multiplication `x = op(A) * x`

.. py:method:: pyculib.blas.Blas.tbmv(uplo, trans, diag, n, k, A, x)

    triangular banded matrix-vector `x = op(A) * x`

.. py:method:: pyculib.blas.Blas.tpmv(uplo, trans, diag, n, AP, x)

    triangular packed matrix-vector multiplication `x = op(A) * x`

.. py:method:: pyculib.blas.Blas.trsv(uplo, trans, diag, n, A, x)

    Solves the triangular linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: pyculib.blas.Blas.tpsv(uplo, trans, diag, n, AP, x)

    Solves the packed triangular linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: pyculib.blas.Blas.tbsv(uplo, trans, diag, n, k, A, x)

    Solves the triangular banded linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: pyculib.blas.Blas.symv(uplo, n, alpha, A, x, beta, y)

    symmetric matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.hemv(uplo, n, alpha, A, x, beta, y)

    Hermitian matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.sbmv(uplo, n, k, alpha, A, x, beta, y)

    symmetric banded matrix-vector multiplication  `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.hbmv(uplo, n, k, alpha, A, x, beta, y)

    Hermitian banded matrix-vector multiplication  `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.spmv(uplo, n, alpha, AP, x, beta, y)

    symmetric packed matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.hpmv(uplo, n, alpha, AP, x, beta, y)

    Hermitian packed matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: pyculib.blas.Blas.ger(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** T + A`

.. py:method:: pyculib.blas.Blas.geru(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** T + A`

.. py:method:: pyculib.blas.Blas.gerc(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** H + A`

.. py:method:: pyculib.blas.Blas.syr(uplo, n, alpha, x, A)

    symmetric rank 1 operation `A := alpha * x * x ** T + A`

.. py:method:: pyculib.blas.Blas.her(uplo, n, alpha, x, A)

    hermitian rank 1 operation  `A := alpha * x * x ** H + A`

.. py:method:: pyculib.blas.Blas.spr(uplo, n, alpha, x, AP)

    the symmetric rank 1 operation `A := alpha * x * x ** T + A`

.. py:method:: pyculib.blas.Blas.hpr(uplo, n, alpha, x, AP)

    hermitian rank 1 operation `A := alpha * x * x ** H + A`

.. py:method:: pyculib.blas.Blas.syr2(uplo, n, alpha, x, y, A)

    symmetric rank-2 update `A = alpha * x * y ** T + y * x ** T + A`

.. py:method:: pyculib.blas.Blas.her2(uplo, n, alpha, x, y, A)

    Hermitian rank-2 update `A = alpha * x * y ** H + alpha * y * x ** H + A`

.. py:method:: pyculib.blas.Blas.spr2(uplo, n, alpha, x, y, A)

    packed symmetric rank-2 update `A = alpha * x * y ** T + y * x ** T + A`

.. py:method:: pyculib.blas.Blas.hpr2(uplo, n, alpha, x, y, A)

    packed Hermitian rank-2 update `A = alpha * x * y ** H + alpha * y * x ** H + A`

BLAS Level 3
-------------

All level 3 routines follow the same naming convention for arguments as in
level 2 routines.

.. py:method:: pyculib.blas.Blas.gemm(transa, transb, m, n, k, alpha, A, B, beta, C)

    matrix-matrix multiplication `C = alpha * op(A) * op(B) + beta * C`

.. py:method:: pyculib.blas.Blas.syrk(uplo, trans, n, k, alpha, A, beta, C)

    symmetric rank- k update `C = alpha * op(A) * op(A) ** T + beta * C`

.. py:method:: pyculib.blas.Blas.herk(uplo, trans, n, k, alpha, A, beta, C)

    Hermitian rank- k update `C = alpha * op(A) * op(A) ** H + beta * C`

.. py:method:: pyculib.blas.Blas.symm(side, uplo, m, n, alpha, A, B, beta, C)

    symmetric matrix-matrix multiplication::

        if  side == 'L':
            C = alpha * A * B + beta * C
        else:  # side == 'R'
            C = alpha * B * A + beta * C

.. py:method:: pyculib.blas.Blas.hemm(side, uplo, m, n, alpha, A, B, beta, C)

    Hermitian matrix-matrix multiplication::

            if  side == 'L':
                C = alpha * A * B + beta * C
            else:   #  side == 'R':
                C = alpha * B * A + beta * C

.. py:method:: pyculib.blas.Blas.trsm(side, uplo, trans, diag, m, n, alpha, A, B)

    Solves the triangular linear system with multiple right-hand-sides::

        if  side == 'L':
            op(A) * X = alpha * B
        else:       # side == 'R'
            X * op(A) = alpha * B


.. py:method:: pyculib.blas.Blas.trmm(side, uplo, trans, diag, m, n, alpha, A, B, C)

    triangular matrix-matrix multiplication::

        if  side == ':'
            C = alpha * op(A) * B
        else:   # side == 'R'
            C = alpha * B * op(A)

.. py:method:: pyculib.blas.Blas.dgmm(side, m, n, A, x, C)

    matrix-matrix multiplication::

        if  mode == 'R':
            C = A * x * diag(X)
        else:       # mode == 'L'
            C = diag(X) * x * A


.. py:method:: pyculib.blas.Blas.geam(transa, transb, m, n, alpha, A, beta, B, C)

    matrix-matrix addition/transposition `C = alpha * op(A) + beta * op(B)`
