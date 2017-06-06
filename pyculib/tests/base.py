from unittest import TestCase, skipIf
from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError


def skip_cuda_tests():

    try:
        if cuda.is_available():
            gpus = cuda.list_devices()
            if gpus and gpus[0].compute_capability >= (2, 0):
                return False
            else:
                return True
        return True
    except CudaSupportError:
        return True


@skipIf(skip_cuda_tests(), "CUDA not supported on this platform.")
class CUDATestCase(TestCase):
    pass
