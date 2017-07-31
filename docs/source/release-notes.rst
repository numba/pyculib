=============
Release notes
=============

Version 1.0.1
=============

Minor documentation and packaging fixes.


Version 1.0.0
=============

NumbaPro and Accelerate have been deprecated, and code generation features have
been moved into open-source Numba. The CUDA library functions have been moved
into Pyculib. There will be no further updates to NumbaPro or Accelerate.

CUDA libraries
--------------

Pyculib CUDA library functionality is equivalent to that in Accelerate 2.+,
with the following packages renamed:

===========================  ===========================
Accelerate package           Pyculib package
===========================  ===========================
``accelerate.cuda.blas``     ``pyculib.blas``
``accelerate.cuda.fft``      ``pyculib.fft``
``accelerate.cuda.rand``     ``pyculib.rand``
``accelerate.cuda.sparse``   ``pyculib.sparse``
``accelerate.cuda.sorting``  ``pyculib.sorting``
===========================  ===========================

