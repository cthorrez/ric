from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_hello.pyx", language_level='3str')
)