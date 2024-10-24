from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "ric",
    sources=["ric.pyx", "src/elo.c"],
    include_dirs=[np.get_include(), "src"],
)

setup(
    name="ric",
    packages=["ric"],
    ext_modules=cythonize([ext]),
    install_requires=["numpy"],
)