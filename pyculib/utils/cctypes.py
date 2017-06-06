from __future__ import absolute_import, print_function, division
from ctypes import c_float, c_double, Structure, c_uint8, sizeof, addressof
import numpy as np


class c_complex(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]

    def __init__(self, real=0, imag=0):
        if isinstance(real, (complex, np.complex64, np.complex128)):
            real, imag = real.real, real.imag
        super(c_complex, self).__init__(real, imag)

    @property
    def value(self):
        return complex(self.real, self.imag)


class c_double_complex(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]

    def __init__(self, real=0, imag=0):
        if isinstance(real, (complex, np.complex64, np.complex128)):
            real, imag = real.real, real.imag
        super(c_double_complex, self).__init__(real, imag)

    @property
    def value(self):
        return complex(self.real, self.imag)


def memalign(cty, align):
    """Allocate a ctype object on the specific byte alignment
    """
    # Allocate bytes with offset
    mem = (c_uint8 * (sizeof(cty) + align))()
    addr = addressof(mem)

    # Move to alignment
    offset = addr % align
    if offset:
        offset = align - offset

    buf = cty.from_address(offset + addr)
    assert 0 == addressof(buf) % align

    return buf, mem
