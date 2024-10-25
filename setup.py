from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "ric",
    sources=["ric.pyx", "src/elo.c"],
    include_dirs=[np.get_include(), "src"],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=["-O3"],  # Set -O3 optimization
    extra_link_args=["-O3"]
)

setup(
    name="ric",
    packages=["ric"],
    ext_modules=cythonize(
        [ext],
        language_level='3',
    ),
    install_requires=["numpy"],
)