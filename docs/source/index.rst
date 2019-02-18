=========
Pyculib
=========

:emphasis:`High Performance Computing`

Pyculib is a package that provides access to several numerical libraries that are optimized for performance on NVidia GPUs.

Pyculib was originally part of Accelerate, developed by Anaconda, Inc.

The current version, 1.0.1, was released on July 27, 2017.

.. note::
    Pyculib is currently available for archival purposes and is not receiving updates.  We think that
    `CuPy <https://cupy.chainer.org/>`_  provides a much more complete interface to standard GPU algorithms, and
    CuPy arrays work with Numba-compiled GPU kernels now.


Features
========

* Bindings to the following :doc:`cuda-libs`:
	* :doc:`cublas`
	* :doc:`cufft`
	* :doc:`cusparse`
	* :doc:`curand`
	* :doc:`sorting` algorithms from the CUB and Modern GPU libraries

Installation
============

This section contains information related to:

.. toctree::
    :maxdepth: 1

    install


User guide
==========

This section contains information related to:

.. toctree::
    :maxdepth: 1

    cuda-libs
    env-variables

Release notes
=============

.. toctree::
   :maxdepth: 1

   release-notes
