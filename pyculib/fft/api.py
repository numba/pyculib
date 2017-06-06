import numpy as np
from . import binding as _cufft
from numba import cuda
from numba.cuda.kernels.transpose import transpose
from numba.cuda.cudadrv.devicearray import DeviceNDArray

def _prepare_types(pairs):
    return dict((tuple(map(np.dtype, k)),
                 getattr(_cufft, 'CUFFT_' + v))
                for k, v in pairs.items())

class FFTPlan(object):
    '''
    :param shape: Input array shape.
    :param itype: Input data type.
    :param otype: Output data type.
    :param batch: Maximum number of operation to perform.
    :param stream: A CUDA stream for all the operations to put on.
    :param mode: Operation mode; e.g. MODE_NATIVE, MODE_FFTW_PADDING,
                 MODE_FFTW_ASYMMETRIC, MODE_FFTW_ALL, MODE_DEFAULT.
    '''

    MODE_NATIVE = _cufft.CUFFT_COMPATIBILITY_NATIVE
    MODE_FFTW_PADDING = _cufft.CUFFT_COMPATIBILITY_FFTW_PADDING
    MODE_FFTW_ASYMMETRIC = _cufft.CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC
    MODE_FFTW_ALL = _cufft.CUFFT_COMPATIBILITY_FFTW_ALL
    MODE_DEFAULT = _cufft.CUFFT_COMPATIBILITY_DEFAULT

    SUPPORTED_TYPES = _prepare_types({
        (np.float32, np.complex64)     : 'R2C',
        (np.float64, np.complex128)    : 'D2Z',
        (np.complex64, np.float32)     : 'C2R',
        (np.complex128, np.float64)    : 'Z2D',
        (np.complex64, np.complex64)   : 'C2C',
        (np.complex128, np.complex128) : 'Z2Z',
    })

    @cuda.require_context
    def __init__(self, shape, itype, otype, batch=1, stream=0,
                 mode=MODE_DEFAULT):

        itype = np.dtype(itype)
        otype = np.dtype(otype)

        try:
            operation = self.SUPPORTED_TYPES[(itype, otype)]
        except KeyError:
            raise ValueError("Invalid type combination")

        if operation in (_cufft.CUFFT_R2C, _cufft.CUFFT_D2Z):
            direction = 'forward'
        elif operation in (_cufft.CUFFT_C2R, _cufft.CUFFT_Z2D):
            direction = 'inverse'
            shape = shape[:-1] + ((shape[-1] - 1) * 2,)
        else:
            direction = 'both'

        self._plan = _cufft.Plan.many(shape, operation, batch=batch)
        if stream:
            self._plan.set_stream(stream)
        self._plan.set_compatibility_mode(mode)

        complex_types = [np.dtype(x) for x in (np.complex64, np.complex128)]

        if itype in complex_types and otype in complex_types:
            ishape = oshape = shape
        elif itype in complex_types:
            ishape = oshape = shape[:-1] + (shape[-1]//2 + 1,)
        elif otype in complex_types:
            ishape = shape
            oshape = shape[:-1] + (shape[-1]//2 + 1,)
        else:
            raise ValueError("Invalid type combination")

        self.operation = operation
        self.itype = itype
        self.otype = otype
        self.shape = shape
        self.ishape = ishape
        self.oshape = oshape
        self.batch = batch
        self.stream = stream
        self.mode = mode
        self.direction = direction

    def _prepare(self, ary, out):
        if ary.shape < self.ishape:
            raise ValueError("Incompatible input array shape")

        if ary.dtype != self.itype:
            raise ValueError("Incompatiable input array dtype")

        do_host_copy = False
        if out is not None:
            h_out = out
            d_out, do_host_copy = cuda._auto_device(out, copy=False,
                                                     stream=self.stream)
        else:
            h_out = np.empty(shape=self.oshape, dtype=self.otype)
            d_out = from_array_like(h_out, self.stream)
            do_host_copy = True

        if h_out.shape < self.oshape:
            raise ValueError("Incompatible output shape")

        d_ary, _ = cuda._auto_device(ary, stream=self.stream)
        return d_ary, d_out, h_out, do_host_copy

    def _maybe_transpose(self, d_ary, d_out):
        """Transpose device arrays into row-major format if needed, as cuFFT
        can't handle column-major data."""

        transpose_in = len(d_ary.shape) == 2 and d_ary.is_f_contiguous()
        transpose_out = len(d_out.shape) == 2 and d_out.is_f_contiguous()
        if transpose_in:
            # Create a row-major device array
            used_in = DeviceNDArray(shape=(d_ary.shape[1], d_ary.shape[0]),
                                    strides=(d_ary.dtype.itemsize,
                                             d_ary.dtype.itemsize*d_ary.shape[1]),
                                    dtype=d_ary.dtype)
            transpose(d_ary, used_in)
        else:
            used_in = d_ary
        if transpose_out:
            # Create a row-major device array
            used_out = DeviceNDArray(shape=d_out.shape,
                                     strides=(d_out.dtype.itemsize*d_out.shape[1],
                                              d_out.dtype.itemsize),
                                     dtype=d_out.dtype)
        else:
            used_out = d_out
        return used_in, used_out, transpose_out
    
    def forward(self, ary, out=None):
        '''Perform forward FFT

        :param ary: Input array
        :param out: Optional output array

        :returns: The output array or a new numpy array is `out` is None.

        .. note:: If `ary` is `out`, an inplace operation is performed.
        '''
        if self.direction not in ('both', 'forward'):
            raise TypeError("Invalid operation")
        d_ary, d_out, h_out, do_host_copy = self._prepare(ary, out)
        used_in, used_out, transpose_out = self._maybe_transpose(d_ary, d_out)
        self._plan.forward(used_in, used_out)
        if do_host_copy:
            if transpose_out:
                # reshape the array so it can be transposed back
                used_out.shape=(used_out.shape[1], used_out.shape[0])
                used_out.strides=(used_out.dtype.itemsize,
                                  used_out.dtype.itemsize*used_out.shape[0])
                # and finally transpose it
                transpose(used_out, d_out)
            d_out.copy_to_host(h_out)
        return h_out

    def inverse(self, ary, out=None):
        '''Perform inverse FFT

        :param ary: Input array
        :param out: Optional output array

        :returns: The output array or a new numpy array is `out` is None.

        .. note: If `ary` is `out`, an inplace operation is performed.
        '''
        if self.direction not in ('both', 'inverse'):
            raise TypeError("Invalid operation")
        d_ary, d_out, h_out, do_host_copy = self._prepare(ary, out)
        used_in, used_out, transpose_out = self._maybe_transpose(d_ary, d_out)
        self._plan.inverse(used_in, used_out)
        if do_host_copy:
            if transpose_out:
                # reshape the array so it can be transposed back
                used_out.shape=(used_out.shape[1], used_out.shape[0])
                used_out.strides=(used_out.dtype.itemsize,
                                  used_out.dtype.itemsize*used_out.shape[0])
                # and finally transpose it
                transpose(used_out, d_out)
            d_out.copy_to_host(h_out)
        return h_out

#
# Simple one-off functions
#

def fft(ary, out, stream=None):
    '''Perform forward FFT on `ary` and output to `out`.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a CUDA stream
    '''
    plan = FFTPlan(ary.shape, ary.dtype, out.dtype, stream=stream)
    plan.forward(ary, out)
    return out

def ifft(ary, out, stream=None):
    '''Perform inverse FFT on `ary` and output to `out`.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a CUDA stream
    '''
    plan = FFTPlan(ary.shape, ary.dtype, out.dtype, stream=stream)
    plan.inverse(ary, out)
    return out

def fft_inplace(ary, stream=None):
    '''Perform inplace forward FFT. `ary` must have complex dtype.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a CUDA stream
    '''
    d_ary, conv = cuda._auto_device(ary, stream=stream)
    fft(d_ary, d_ary, stream=stream)
    if conv:
        d_ary.copy_to_host(ary)
    return ary


def ifft_inplace(ary, stream=None):
    '''Perform inplace inverse FFT. `ary` must have complex dtype.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a CUDA stream
    '''
    d_ary, conv = cuda._auto_device(ary, stream=stream)
    ifft(d_ary, d_ary, stream=stream)
    if conv:
        d_ary.copy_to_host(ary)
    return ary
