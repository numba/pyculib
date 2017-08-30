#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.spawn import spawn
from distutils.command import build

import versioneer

class build_doc(build.build):
    description = "build documentation"

    def run(self):
        spawn(['make', '-C', 'docs', 'html'])

packages = [
    'pyculib',
    'pyculib.blas',
    'pyculib.fft',
    'pyculib.rand',
    'pyculib.sparse',
    'pyculib.sorting',
    'pyculib.utils',
    'pyculib.tests',
]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_doc'] = build_doc

with open('README.rst', 'r') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(
        name='pyculib',
        description='Pyculib - python bindings for NVIDIA CUDA libraries',
        long_description=long_description,
        author='Anaconda, Inc.',
        author_email='support@anaconda.com',
        url='https://anaconda.com',
        packages=packages,
        license='BSD',
        version=versioneer.get_version(),
        cmdclass=cmdclass,
    )
