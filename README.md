# Pyculib

Pyculib provides Python bindings to the following CUDA libraries:

 * [cuBLAS](https://developer.nvidia.com/cublas)
 * [cuFFT](https://developer.nvidia.com/cufft)
 * [cuSPARSE](https://developer.nvidia.com/cusparse)
 * [cuRAND](https://developer.nvidia.com/curand)
 * CUDA Sorting algorithms from the [CUB](https://nvlabs.github.io/cub/) and
   [Modern GPU](https://github.com/moderngpu/moderngpu) libraries.

These bindings are direct ports of those available in [Anaconda
Accelerate](https://docs.continuum.io/accelerate/cuda-libs).

Documentation is located [here](http://pyculib.readthedocs.io/en/latest/)

## Installing

The easiest way to install Pyculib and get updates is by using the [Anaconda
Distribution](https://www.continuum.io/downloads)

```
#> conda install pyculib
```

To compile from source, it is recommended to create a conda environment
containing the following:

 * cffi
 * cudatoolkit
 * numpy
 * numba
 * pyculib\_sorting
 * scipy

for instructions on how to do this see the [conda](https://conda.io/docs/)
documentation, specifically the section on [managing
environments](https://conda.io/docs/using/envs.html#managing-environments).

Once a suitable environment is activated, installation achieved simply by
running:

```
#> python setup.py install
```

and the installation can be tested with:

```
#> ./runtests.py
```


## Documentation

Documentation is located [here](http://pyculib.readthedocs.io/en/latest/).

### Building Documentation

It is also possible to build a local copy of the documentation from source.
This requires GNU Make and sphinx (available via conda).


Documentation is stored in the `doc` folder, and should be built with:

```
#> make SPHINXOPTS=-Wn clean html
```

This ensures that the documentation renders without errors. If errors occur,
they can all be seen at once by building with:

```
#> make SPHINXOPTS=-n clean html
```

However, these errors should all be fixed so that building with `-Wn` is
possible prior to merging any documentation changes or updates.

