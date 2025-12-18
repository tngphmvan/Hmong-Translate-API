
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Force the extension name to be just "core" so it builds in the current folder
extensions = [
    Extension("core", ["core.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='monotonic_align',
    ext_modules=cythonize(extensions),
)
