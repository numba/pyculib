cuSPARSE
========

Provides basic linear algebra operations for sparse matrices. See `NVIDIA
cuSPARSE <http://docs.nvidia.com/cuda/cusparse/>`_ for an in-depth description
of the cuSPARSE library and its methods and data types. All functions are
accessed through the :class:`pyculib.sparse.Sparse` class:

.. autoclass:: pyculib.sparse.Sparse

Similarly to the cuBLAS interface, no special naming convention is used for
functions to operate on different datatypes - all datatypes are handled by each
function, and dispatch of the corresponding library function is handled by
Pyculib. However, it is often necessary to provide a *matrix descriptor* to
functions, which provides some information about the format and properties of a
matrix. A matrix descriptor can be obtained from the
:py:meth:`pyculib.sparse.Sparse.matdescr` method:

.. py:method:: pyculib.sparse.Sparse.matdescr(indexbase, diagtype, fillmode, matrixtype)

   Creates a matrix descriptor that describes a matrix with the given
   `indexbase`, `diagtype`, `fillmode`, and `matrixtype`. Note that not all of
   these options are relevant to every matrix storage format.

   :param indexbase: Optional. 0 for 0-based indexing, or 1 for 1-based
                     indexing. If not specified, the default given to the
                     :py:class:`pyculib.sparse.Sparse` constructor is
                     used instead.
   :param diagtype: Optional. Defaults to `'N'`. `'N'` signifies that the matrix
                    diagonal has non-unit elements. `'U'` signifies that the
                    matrix diagonal only contains unit elements.
   :param fillmode: Optional. Defaults to `'L'`. `'L'` indicates that the lower
                    triangular part of the matrix is stored. `'U'` indicates
                    that the upper triangular part of the matrix is stored.
   :param matrixtype: Optional. Defaults to `'G'`. `'S'` indicates that the
                      matrix is symmetric. `'H'` indicates that the matrix is
                      Hermitian. `'T'` indicates that the matrix is triangular.
                      `'G'` is used for a *general* matrix, which is not
                      symmetric, Hermitian, or triangular.
   :return: A matrix descriptor.

Many of the methods of the :class:`pyculib.sparse.Sparse` class accept
the individual data structures that make up a sparse representation of a matrix
(for example the values, the row pointers and the column indices for a CSR
format matrix). However, some methods (such as
:py:meth:`pyculib.sparse.Sparse.csrgemm_ez`), accept an instance of the
:class:`pyculib.sparse.CudaSparseMatrix` class:

.. py:class:: pyculib.sparse.CudaSparseMatrix()

   Base class for a representation of a sparse matrix on a CUDA device. The
   constructor takes no arguments.

   .. py:method:: from_host_matrix(matrix, stream)

      Initialise the matrix structure and values from an instance of a matrix on
      the host. The host matrix must be of the corresponding host type, which is
      documented for each subclass below.

   .. py:method:: copy_to_host(stream)

      Create an instance of the corresponding host matrix type and copy the
      matrix structure and data into it from the device. See subclass
      documentation for an indication of the corresponding matrix type.

Subclasses of the sparse matrix type are:

.. py:class:: pyculib.sparse.CudaBSRMatrix()

   CUDA sparse matrix for which the corresponding type is a
   :py:class:`scipy.sparse.bsr_matrix`.

.. py:class:: pyculib.sparse.CudaCSRMatrix()

   CUDA sparse matrix for which the corresponding type is a
   :py:class:`scipy.sparse.csr_matrix`.

.. py:class:: pyculib.sparse.CudaCSCMatrix()

   CUDA sparse matrix for which the corresponding type is a
   :py:class:`scipy.sparse.csc_matrix`.

There are also some convenience methods for constructing CUDA sparse matrices in
a similar manner to Scipy sparse matrices:

.. automethod:: pyculib.sparse.bsr_matrix

.. automethod:: pyculib.sparse.csr_matrix

.. automethod:: pyculib.sparse.csc_matrix

BLAS Level 1
------------

.. py:method:: pyculib.sparse.Sparse.axpyi(alpha, xVal, xInd, y)

   Multiplies the sparse vector `x` by `alpha` and adds the result to the dense
   vector `y`.

   :param alpha: scalar
   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: dense vector

.. py:method:: pyculib.sparse.Sparse.doti(xVal, xInd, y)

   Computes the dot product of the sparse vector `x` and dense vector `y`.

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: scalar

.. py:method:: pyculib.sparse.Sparse.dotci(xVal, xInd, y)

   Computes the dot product of the complex conjugate of the sparse vector `x`
   and the dense vector `y`.

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: scalar

.. py:method:: pyculib.sparse.Sparse.gthr(y, xVal, xInd)

   Gathers the elements of `y` at the indices `xInd` into the array `xVal`

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.gthrz(y, xVal, xInd)

   Gathers the elements of `y` at the indices `xInd` into the array `xVal` and
   zeroes out the gathered elements of `y`.

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.roti(xVal, xInd, y, c, s)

   Applies the Givens rotation matrix, `G`:

   .. math::

      G = \left( \begin{array}{cc}
      C  & S \\
      -S & C
      \end{array}\right)

   to the sparse vector `x` and dense vector
   `y`.

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :param c: cosine element of the rotation matrix
   :param s: sine element of the rotation matrix
   :return: None

.. py:method:: pyculib.sparse.Sparse.sctr(xVal, xInd, y)

   Scatters the elements of the sparse vector `x` into the dense vector `y`.
   Elements of `y` whose indices are not listed in `xInd` are unmodified.

   :param xVal: vector of non-zero values of `x`
   :param xInd: vector of indices of non-zero values of `x`
   :param y: dense vector
   :return: None


BLAS Level 2
------------

All level 2 routines follow the following naming convention for the following
arguments:

* alpha, beta -- (scalar) Can be real or complex numbers.
* descr, descrA, descrB -- (descriptor) Matrix descriptor. An appropriate
  descriptor may be obtained by calling
  :py:meth:`pyculib.sparse.Sparse.matdescr`. `descr` only applies to the
  only matrix argument. `descrA` and `descrB` apply to matrix `A` and matrix
  `B`, respectively.
* dir -- (string) Can be `'C'` to indicate column-major block storage or `'R'`
  to indicate row-major block storage.
* trans, transa, transb -- (string)
                Select the operation `op` to apply to a matrix:

                - `'N'`: `op(X) = X`, the identity operation;
                - `'T'`: `op(X) = X**T`, the transpose;
                - `'C'`: `op(X) = X**H`, the conjugate transpose.

                `trans` only applies to the only matrix argument.
                `transa` and `transb` apply to matrix `A` and matrix `B`,
                respectively.


.. py:method:: pyculib.sparse.Sparse.bsrmv_matrix(dir, trans, alpha, descr, bsrmat, x, beta, y)

   Matrix-vector multiplication `y = alpha * op(A) * x + beta * y` with a
   BSR-format matrix.

   :param dir: block storage direction
   :param trans: operation to apply to the matrix
   :param alpha: scalar
   :param descr: matrix descriptor
   :param bsrmat: the matrix `A`
   :param x: dense vector
   :param beta: scalar
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.bsrmv(dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y)

   Matrix-vector multiplication `y = alpha * op(A) * x + beta * y` with a
   BSR-format matrix. This function accepts the individual arrays that make up
   the structure of a BSR matrix - if a
   :class:`pyculib.sparse.CudaBSRMatrix` instance is to hand, it is
   recommended to use the :py:meth:`bsrmv_matrix` method instead.

   :param dir: block storage direction
   :param trans: operation to apply to the matrix
   :param mb: Number of block rows of the matrix
   :param nb: Number of block columns of the matrix
   :param nnzb: Number of nonzero blocks of the matrix
   :param alpha: scalar
   :param descr: matrix descriptor
   :param bsrVal: vector of nonzero values of the matrix
   :param bsrRowPtr: vector of block row pointers of the matrix
   :param bsrColInd: vector of block column indices of the matrix
   :param blockDim: block dimension of the matrix
   :param x: dense vector
   :param beta: scalar
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.bsrxmv(dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y)

   Matrix-vector multiplication similar to :py:meth:`bsrmv`, but including a
   mask operation: `y(mask) = (alpha * op(A) * x + beta * y)(mask)`. The blocks
   of y to be updated are specified in `bsrMaskPtr`. Blocks whose indices are
   not specified in `bsrMaskPtr` are left unmodified.

   :param dir: block storage direction
   :param trans: operation to apply to the matrix
   :param sizeOfMask: number of updated blocks of rows of `y`
   :param mb: Number of block rows of the matrix
   :param nb: Number of block columns of the matrix
   :param nnzb: Number of nonzero blocks of the matrix
   :param alpha: scalar
   :param descr: matrix descriptor
   :param bsrVal: vector of nonzero values of the matrix
   :param bsrMaskPtr: vector of indices of the block elements to be updated
   :param bsrRowPtr: vector of block row pointers of the matrix
   :param bsrEndPtr: vector of pointers to the end of every block row plus one
   :param bsrColInd: vector of block column indices of the matrix
   :param blockDim: block dimension of the matrix
   :param x: dense vector
   :param beta: scalar
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrmv(trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y)

   Matrix-vector multiplication `y = alpha * op(A) * x + beta * y` with a
   CSR-format matrix.

   :param trans: operation to apply to the matrix
   :param m: Number of rows of the matrix
   :param n: Number of columns of the matrix
   :param nnz: Number of nonzeroes of the matrix
   :param alpha: scalar
   :param descr: matrix descriptor
   :param csrVal: vector of nonzero values of the matrix
   :param csrRowPtr: vector of row pointers of the matrix
   :param csrColInd: vector of column indices of the matrix
   :param x: dense vector
   :param beta: scalar
   :param y: dense vector
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrsv_analysis(trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd)

   Performs the analysis phase of the solution of the sparse triangular linear
   system `op(A) * y = alpha * x`. This needs to be executed only once for a
   given matrix and operation type.

   :param trans: operation to apply to the matrix
   :param m: number of rows of the matrix
   :param nnz: number of nonzeroes of the matrix
   :param descr: matrix descriptor
   :param csrVal: vector of nonzero values of the matrix
   :param csrRowPtr: vector of row pointers of the matrix
   :param csrColInd: vector of column indices of the matrix
   :return: the analysis result, which can be used as input to the solve phase

.. py:method:: pyculib.sparse.Sparse.csrsv_solve(trans, m, alpha, descr, csrVal, csrRowPtr, csrColInd, info, x, y)

   Performs the analysis phase of the solution of the sparse triangular linear
   system `op(A) * y = alpha * x`.

   :param trans: operation to apply to the matrix
   :param m: number of rows of the matrix
   :param alpha: scalar
   :param descr: matrix descriptor
   :param csrVal: vector of nonzero values of the matrix
   :param csrRowPtr: vector of row pointers of the matrix
   :param csrColInd: vector of column indices of the matrix
   :param info: the analysis result from :py:meth:`csrsv_analysis`
   :param x: dense vector
   :param y: dense vector into which the solve result is stored
   :return: None


BLAS Level 3
------------

.. py:method:: pyculib.sparse.Sparse.csrmm(transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)

   Matrix-matrix multiplication `C = alpha * op(A) * B + beta * C` where `A` is
   a sparse matrix in CSR format and `B` and `C` are dense matrices.

   :param transA: operation to apply to `A`
   :param m: number of rows of `A`
   :param n: number of columns of `B` and `C`
   :param k: number of columns of `A`
   :param nnz: number of nonzeroes in `A`
   :param alpha: scalar
   :param descrA: matrix descriptor
   :param csrValA: vector of nonzero values of `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param B: dense matrix
   :param ldb: leading dimension of `B`
   :param beta: scalar
   :param C: dense matrix
   :param ldc: leading dimension of `C`
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrmm2(transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)

   Matrix-matrix multiplication `C = alpha * op(A) * op(B) + beta * C` where `A` is
   a sparse matrix in CSR format and `B` and `C` are dense matrices.

   :param transA: operation to apply to `A`
   :param transB: operation to apply to `B`
   :param m: number of rows of `A`
   :param n: number of columns of `B` and `C`
   :param k: number of columns of `A`
   :param nnz: number of nonzeroes in `A`
   :param alpha: scalar
   :param descrA: matrix descriptor
   :param csrValA: vector of nonzero values of `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param B: dense matrix
   :param ldb: leading dimension of `B`
   :param beta: scalar
   :param C: dense matrix
   :param ldc: leading dimension of `C`
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrsm_analysis(transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA)

   Performs the analysis phase of the solution of a sparse triangular linear
   system `op(A) * Y = alpha * X` with multiple right-hand sides where `A` is a
   sparse matrix in CSR format, and `X` and `Y` are dense matrices.

   :param transA: operation to apply to `A`
   :param m: number of rows of `A`
   :param nnz: number of nonzeroes in `A`
   :param descrA: matrix descriptor
   :param csrValA: vector of nonzero values of `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :return: the analysis result

.. py:method:: pyculib.sparse.Sparse.csrsm_solve(transA, m, n, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, X, ldx, Y, ldy)

   Performs the analysis phase of the solution of a sparse triangular linear
   system `op(A) * Y = alpha * X` with multiple right-hand sides where `A` is a
   sparse matrix in CSR format, and `X` and `Y` are dense matrices.

   :param transA: operation to apply to `A`
   :param m: number of rows of `A`
   :param n: number of columns of `B` and `C`
   :param alpha: scalar
   :param descrA: matrix descriptor
   :param csrValA: vector of nonzero values of `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param info: the analysis result from :py:meth:`csrsm_analysis`
   :param X: dense matrix
   :param ldx: leading dimension of `X`
   :param Y: dense matrix
   :param ldy: leading dimension of `Y`
   :return: None


Extra Functions
---------------

.. py:method:: pyculib.sparse.Sparse.XcsrgeamNnz(m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC)

   Set up the sparsity pattern for the matrix operation `C = alpha * A + beta *
   B` where `A`, `B`, and `C` are all sparse matrices in CSR format.

   :param m: number of rows of all matrices
   :param n: number of columns of all matrices
   :param descrA: matrix descriptor for `A`
   :param nnzA: number of nonzeroes in `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param descrB: matrix descriptor for `B`
   :param nnzB: number of nonzeroes in `B`
   :param csrRowPtrB: vector of row pointers of `B`
   :param csrColIndB: vector of column indices of `B`
   :param descrC: matrix descriptor for `B`
   :param csrRowPtrC: vector of row pointers of `C`, written to by this method
   :return: number of nonzeroes in `C`

.. py:method:: pyculib.sparse.Sparse.csrgeam(m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)

   Performs the the matrix operation `C = alpha * A + beta * B` where `A`, `B`,
   and `C` are all sparse matrices in CSR format.

   :param m: number of rows of all matrices
   :param n: number of columns of all matrices
   :param alpha: scalar
   :param descrA: matrix descriptor for `A`
   :param nnzA: number of nonzeroes in `A`
   :param csrValA: vector of nonzero values of `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param beta: scalar
   :param descrB: matrix descriptor for `B`
   :param nnzB: number of nonzeroes in `B`
   :param csrValB: vector of nonzero values of `B`
   :param csrRowPtrB: vector of row pointers of `B`
   :param csrColIndB: vector of column indices of `B`
   :param descrC: matrix descriptor for `B`
   :param csrValC: vector of nonzero values of `C`
   :param csrRowPtrC: vector of row pointers of `C`
   :param csrColIndC: vector of column indices of `C`
   :return: None

.. py:method:: pyculib.sparse.Sparse.XcsrgemmNnz(transA, transB, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC)

   Set up the sparsity pattern for the matrix operation `C = op(A) * op(B)`
   where `A`, `B`, and `C` are all sparse matrices in CSR format.

   :param transA: operation to apply to `A`
   :param transB: operation to apply to `B`
   :param m: number of rows of `A` and `C`
   :param n: number of columns of `B` and `C`
   :param k: number of columns/rows of `A`/`B`
   :param descrA: matrix descriptor for `A`
   :param nnzA: number of nonzeroes in `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param descrB: matrix descriptor for `B`
   :param nnzB: number of nonzeroes in `B`
   :param csrRowPtrB: vector of row pointers of `B`
   :param csrColIndB: vector of column indices of `B`
   :param descrC: matrix descriptor for `C`
   :param csrRowPtrC: vector of row pointers of `C`, written by this function
   :return: number of nonzeroes in `C`

.. py:method:: pyculib.sparse.Sparse.csrgemm(transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)

   Perform the matrix operation `C = op(A) * op(B)` where `A`, `B`, and `C` are
   all sparse matrices in CSR format.

   :param transA: operation to apply to `A`
   :param transB: operation to apply to `B`
   :param m: number of rows of `A` and `C`
   :param n: number of columns of `B` and `C`
   :param k: number of columns/rows of `A`/`B`
   :param descrA: matrix descriptor for `A`
   :param nnzA: number of nonzeroes in `A`
   :param csrValA: vector of nonzero values in `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param descrB: matrix descriptor for `B`
   :param nnzB: number of nonzeroes in `B`
   :param csrValB: vector of nonzero values in `B`
   :param csrRowPtrB: vector of row pointers of `B`
   :param csrColIndB: vector of column indices of `B`
   :param descrC: matrix descriptor for `C`
   :param csrValC: vector of nonzero values in `C`
   :param csrRowPtrC: vector of row pointers of `C`
   :param csrColIndC: vector of column indices of `C`
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrgemm_ez(A, B, transA='N', transB='N', descrA=None, descrB=None, descrC=None)

    Performs the matrix operation `C = op(A) * op(B)` where `A`, `B` and `C`
    are all sparse matrices in CSR format. This function accepts and returns
    :py:class:`pyculib.sparse.CudaCSRMatrix` matrices, and makes
    calls to :py:meth:`XcsrgemmNnz` and :py:meth:`csrgemm`.

   :param A: :py:class:`pyculib.sparse.CudaCSRMatrix`
   :param B: :py:class:`pyculib.sparse.CudaCSRMatrix`
   :param transA: optional, operation to apply to `A`
   :param transB: optional, operation to apply to `B`
   :param descrA: optional, matrix descriptor for `A`
   :param descrB: optional, matrix descriptor for `B`
   :param descrC: optional, matrix descriptor for `C`
   :return: :py:class:`pyculib.sparse.CudaCSRMatrix`


Preconditioners
---------------

.. py:method:: pyculib.sparse.Sparse.csric0(trans, m, descr, csrValA, csrRowPtrA, csrColIndA, info)

   Computes incomplete Cholesky factorization of a sparse matrix in CSR format
   with 0 fill-in and no pivoting: `op(A) = R**T * R`. This method must follow a
   call to :py:meth:`csrsv_analysis`. The matrix `A` is overwritten with the
   upper or lower triangular factors `R` or `R**T`.

   :param trans: operation to apply to the matrix
   :param m: number of rows and columns of the matrix
   :param descr: matrix descriptor
   :param csrValA: vector of nonzero values in `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param info: analysis result
   :return: None

.. py:method:: pyculib.sparse.Sparse.csrilu0(trans, m, descr, csrValA, csrRowPtrA, csrColIndA, info)

   Computes incomplete-LU factorization of a sparse matrix in CSR format with 0
   fill-in and no pivoting: `op(A) = L * U`. This method must follow a call to
   :py:meth:`csrsv_analysis`. The matrix `A` is overwritten with the lower and
   upper triangular factors `L` and `U`.

   :param trans: operation to apply to the matrix
   :param m: number of rows and columns of the matrix
   :param descr: matrix descriptor
   :param csrValA: vector of nonzero values in `A`
   :param csrRowPtrA: vector of row pointers of `A`
   :param csrColIndA: vector of column indices of `A`
   :param info: analysis result
   :return: None

.. py:method:: pyculib.sparse.Sparse.gtsv(m, n, dl, d, du, B, ldb)

   Computes the solution of a tridiagonal linear system with multiple right-hand
   sides: `A * Y = alpha * X`.

   :param m: the size of the linear system
   :param n: the number of right-hand sides in the system
   :param dl: dense vector storing the lower-diagonal elements
   :param d: dense vector storing the diagonal elements
   :param du: dense vector storing the upper-diagonal elements
   :param B: dense matrix holding the right-hand sides of the system
   :param ldb: the leading dimension of `B`
   :return: None

.. py:method:: pyculib.sparse.Sparse.gtsv_nopivot(m, n, dl, d, du, B, ldb)

   Similar to :py:meth:`gtsv`, but computes the solution without performing any
   pivoting.

   :param m: the size of the linear system
   :param n: the number of right-hand sides in the system
   :param dl: dense vector storing the lower-diagonal elements
   :param d: dense vector storing the diagonal elements
   :param du: dense vector storing the upper-diagonal elements
   :param B: dense matrix holding the right-hand sides of the system
   :param ldb: the leading dimension of `B`
   :return: None

.. py:method:: pyculib.sparse.Sparse.gtsvStridedBatch(m, dl, d, du, x, batchCount, batchStride)

   Computes the solution of `i` tridiagonal linear systems: `A(i) * y(i) = alpha
   * x(i)`.

   :param m: the size of the linear systems
   :param dl: stacked dense vector storing the lower-diagonal elements of each
              system
   :param d: stacked dense vector storing the diagonal elements of each system
   :param du: stacked dense vector storing the upper-diagonal elements of each
              system
   :param x: dense matrix holding the right-hand sides of the systems
   :param batchCount: number of systems to solve
   :param batchStride: number of elements separating the vectors of each system
   :return: None


Format Conversion
-----------------

.. py:method:: pyculib.sparse.Sparse.bsr2csr(dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC)

   Convert the sparse matrix `A` in BSR format to CSR format, stored in `C`.

   :param dirA: row ('R') or column ('C') orientation of block storage
   :param mb: number of block rows of `A`
   :param nb: number of block columns of `A`
   :param descrA: matrix descriptor for `A`
   :param bsrValA: vector of nonzero values of `A`
   :param bsrRowPtrA: vector of block row pointers of `A`
   :param bsrColIndA: vector of block column indices of `A`
   :param blockDim: block dimension of `A`
   :param descrC: matrix descriptor for `C`
   :param csrValA: vector of nonzero values in `C`
   :param csrRowPtrA: vector of row pointers of `C`
   :param csrColIndA: vector of column indices of `C`
   :return: None

.. py:method:: pyculib.sparse.Sparse.Xcoo2csr(cooRowInd, nnz, m, csrRowPtr)

   Converts an array containing uncompressed row indices corresponding to the
   COO format into into an array of compressed row pointers corresponding to the
   CSR format.

   :param cooRowInd: integer array of uncompressed row indices
   :param nnz: number of nonzeroes
   :param m: number of matrix rows
   :param csrRowPtr: vector of row pointers to be written to
   :return: None

.. py:method:: pyculib.sparse.Sparse.csc2dense(m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)

   Convert the sparse matrix `A` in CSC format into a dense matrix.

   :param m: number of rows of `A`
   :param n: number of columns of `A`
   :param descrA: matrix descriptor for `A`
   :param cscValA: values in the CSC representation of `A`
   :param cscRowIndA: row indices in the CSC representation of `A`
   :param cscColPtrA: column pointers in the CSC representation of `A`
   :param A: dense matrix representation of `A` to be written by this function.
   :param lda: leading dimension of `A`
   :return: None

.. py:method:: pyculib.sparse.Sparse.Xcsr2bsrNnz(dirA, m, n, descrA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrRowPtrC)

   Performs the analysis necessary for converting a matrix in CSR format into
   BSR format.

   :param dirA: row ('R') or column ('C') orientation of block storage
   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descrA: matrix descriptor for input matrix `A`
   :param csrRowPtrA: row pointers of matrix
   :param csrColIndA: column indices of matrix
   :param blockDim: block dimension of output matrix `C`
   :param descrC: matrix descriptor for output matrix `C`
   :return: number of nonzeroes of matrix

.. py:method:: pyculib.sparse.Sparse.csr2bsr(dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC)

   Performs conversion of a matrix from CSR format into BSR format.

   :param dirA: row ('R') or column ('C') orientation of block storage
   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descrA: matrix descriptor for input matrix `A`
   :param csrValA: nonzero values of matrix
   :param csrRowPtrA: row pointers of matrix
   :param csrColIndA: column indices of matrix
   :param blockDim: block dimension of output matrix `C`
   :param descrC: matrix descriptor for output matrix `C`
   :param bsrValC: nonzero values of output matrix `C`
   :param bsrRowPtrC: block row pointers of output matrix `C`
   :param bsrColIndC: block column indices of output matrix `C`
   :return: number of nonzeroes of matrix

.. py:method:: pyculib.sparse.Sparse.Xcsr2coo(csrRowPtr, nnz, m, cooRowInd)

   Converts an array of compressed row pointers corresponding to the CSR format
   into an array of uncompressed row indices corresponding to the COO format.

   :param csrRowPtr: vector of row pointers
   :param nnz: number of nonzeroes
   :param m: number of rows of matrix
   :param cooRowInd: vector of uncompressed row indices written by this function
   :return: None

.. py:method:: pyculib.sparse.Sparse.csr2csc(m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues)

   Converts a sparse matrix in CSR format into a sparse matrix in CSC format.

   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param nnz: number of nonzeroes of the matrix
   :param csrVal: values in the CSR representation
   :param csrRowPtr: row indices in the CSR representation
   :param csrColInd: column pointers in the CSR representation
   :param cscVal: values in the CSC representation
   :param cscRowInd: row indices in the CSC representation
   :param cscColPtr: column pointers in the CSC representation
   :param copyValues: `'N'` or `'S'` for symbolic or numeric copy of values
   :return: None

.. py:method:: pyculib.sparse.Sparse.csr2dense(m, n, descr, csrVal, csrRowPtr, csrColInd, A, lda)

   Convert a sparse matrix in CSR format into dense format.

   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descr: matrix descriptor
   :param csrVal: values in the CSR representation
   :param csrRowPtr: row indices in the CSR representation
   :param csrColInd: column pointers in the CSR representation
   :param A: the dense representation, written to by this function
   :param lda: leading dimension of the matrix
   :return: None

.. py:method:: pyculib.sparse.Sparse.dense2csc(m, n, descrA, A, lda, nnzPerCol, cscVal, cscRowInd, cscColPtr)

   Convert a dense matrix into a sparse matrix in CSC format. The `nnzPerCol`
   parameter may be computed with a call to :py:meth:`nnz`.

   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descrA: matrix descriptor
   :param A: the matrix in dense format
   :param lda: leading dimension of the matrix
   :param nnzPerCol: array containing the number of nonzero elements per column
   :param cscVal: values in the CSC representation
   :param cscRowInd: row indices in the CSC representation
   :param cscColPtr: column pointers in the CSC representation
   :return: None

.. py:method:: pyculib.sparse.Sparse.dense2csr(m, n, descrA, A, lda, nnzPerRow, csrVal, csrRowPtr, csrColInd)

   Convert a dense matrix into a sparse matrix in CSR format. The `nnzPerRow`
   parameter may be computed with a call to :py:meth:`nnz`.

   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descrA: matrix descriptor
   :param A: the matrix in dense format
   :param lda: leading dimension of the matrix
   :param nnzPerRow: array containing the number of nonzero elements per row
   :param csrVal: values in the CSR representation
   :param csrRowPtr: row indices in the CSR representation
   :param csrColInd: column pointers in the CSR representation
   :return: None

.. py:method:: pyculib.sparse.Sparse.nnz(dirA, m, n, descrA, A, lda, nnzPerRowCol)

   Computes the number of nonzero elements per row or column of a dense matrix,
   and the total number of nonzero elements in the matrix.

   :param dirA: `'R'` for the number of nonzeroes per row, or `'C'` for per
                column.
   :param m: number of rows of matrix
   :param n: number of columns of matrix
   :param descrA: matrix descriptor
   :param A: the matrix
   :param lda: leading dimension of the matrix
   :param nnzPerRowCol: array to contain the number of nonzeroes per row or
                        column
   :return: total number of nonzeroes in the matrix
