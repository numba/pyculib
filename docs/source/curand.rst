cuRAND
======

Provides `pseudo-random number generator` (PRNG) and `quasi-random generator` (QRNG).
See `NVIDIA cuRAND <http://docs.nvidia.com/cuda/curand/index.html>`_.

class PRNG
-----------

.. autoclass:: pyculib.rand.PRNG
   :members:


class QRNG
------------

.. autoclass:: pyculib.rand.QRNG
   :members:


Top Level PRNG Functions
--------------------------

Simple interface to the PRNG methods.

.. note:: This methods automatically create a PRNG object.

.. autofunction:: pyculib.rand.uniform

.. autofunction:: pyculib.rand.normal

.. autofunction:: pyculib.rand.lognormal

.. autofunction:: pyculib.rand.poisson

Top Level QRNG Functions
--------------------------

Simple interface to the QRNG methods.

.. note:: This methods automatically create a QRNG object.

.. autofunction:: pyculib.rand.quasi
