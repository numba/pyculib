# Pyculib QuickStart

Pyculib provides access to NVIDIA' optimized CUDA libraries
from a high-level, Pythonic interface. It builds on top of the functionality
provided in the open-source Numba JIT compiler.


## How do I install it?

System requirements:

* Python 2.7, 3.4+
* Numpy 1.10 or later
* NVIDIA CUDA-enabled GPU with compute
  capability 2.0 or above. CUDA Toolkit 7.5 and driver version 349.00 or above
  ([https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit))
* Numba 0.33+

### Install from Anaconda

Download and install Anaconda from
[https://www.anaconda.com/download](https://www.anaconda.com/download).

In terminal:

```
conda update conda
conda install pyculib
```

## How do I use it?

Pyculib provides access to optimized dense and sparse linear algebra, random
number generators, sorting routines, and FFTs. This example demonstrates the use
of CUDA-FFT routines:


### CUDA-Accelerated FFT

```python
from pyculib.fft import fft
fft(x, xf)
```

## Where do I learn more?

* Full documentation: (Insert URL)
* CUDA-accelerated functions: (Insert URL)/cudalibs.html
