**NOTE**: This project is no longer receiving updates, as we think that CuPy is provides a much more complete interface to standard GPU algorithms.  CuPy arrays also work with Numba-compiled GPU kernels now.  We encourage you to take a look at CuPy:

https://cupy.chainer.org/


Pyculib
=======

Pyculib provides Python bindings to the following CUDA libraries:

-  `cuBLAS <https://developer.nvidia.com/cublas>`__
-  `cuFFT <https://developer.nvidia.com/cufft>`__
-  `cuSPARSE <https://developer.nvidia.com/cusparse>`__
-  `cuRAND <https://developer.nvidia.com/curand>`__
-  CUDA Sorting algorithms from the
   `CUB <https://nvlabs.github.io/cub/>`__ and `Modern
   GPU <https://github.com/moderngpu/moderngpu>`__ libraries.

These bindings are direct ports of those available in `Anaconda
Accelerate <https://docs.continuum.io/accelerate/cuda-libs>`__.

Documentation is located
`here <http://pyculib.readthedocs.io/en/latest/>`__

Installing
----------

The easiest way to install Pyculib and get updates is by using the
`Anaconda Distribution <https://www.anaconda.com/download>`__

::

    #> conda install pyculib

To compile from source, it is recommended to create a conda environment
containing the following:

-  cffi
-  cudatoolkit
-  numpy
-  numba
-  pyculib\_sorting
-  scipy

for instructions on how to do this see the
`conda <https://conda.io/docs/>`__ documentation, specifically the
section on `managing
environments <https://conda.io/docs/using/envs.html#managing-environments>`__.

Once a suitable environment is activated, installation achieved simply
by running:

::

    #> python setup.py install

and the installation can be tested with:

::

    #> ./runtests.py

Documentation
-------------

Documentation is located
`here <http://pyculib.readthedocs.io/en/latest/>`__.

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

It is also possible to build a local copy of the documentation from
source. This requires GNU Make and sphinx (available via conda).

Documentation is stored in the ``doc`` folder, and should be built with:

::

    #> make SPHINXOPTS=-Wn clean html

This ensures that the documentation renders without errors. If errors
occur, they can all be seen at once by building with:

::

    #> make SPHINXOPTS=-n clean html

However, these errors should all be fixed so that building with ``-Wn``
is possible prior to merging any documentation changes or updates.
