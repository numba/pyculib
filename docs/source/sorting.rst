============
CUDA Sorting
============

Pyculib provides routines for sorting arrays on CUDA GPUs.

Sorting Large Arrays
====================

The :py:class:`pyculib.sorting.RadixSort` class is recommended for
sorting large (approx. more than 1 million items) arrays of numeric types.

.. autoclass:: pyculib.sorting.RadixSort
   :members:

Sorting Many Small Arrays
=========================

Using :py:class:`pyculib.sorting.RadixSort` on small (approx. less than
1 million items) arrays has significant overhead due to multiple kernel
launches. 

A better alternative is to use :py:func:`pyculib.sorting.segmented_sort`-which launches a single kernel for sorting a batch of many small arrays.

.. autofunction:: pyculib.sorting.segmented_sort
