from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "ric",
    sources=[
        "ric.pyx",
        "src/elo.c",
        "src/glicko.c",
        # "src/trueskill.c",
        # "src/eval.c"
    ],
    include_dirs=[np.get_include(), "src"],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-mavx",
        "-mavx2",
        "-mfma",
        "-ffast-math"
    ],
    extra_link_args=["-lm"]
)

setup(
    name="ric",
    packages=["ric"],
    ext_modules=cythonize(
        [ext],
        language_level='3',
        compiler_directives={'embedsignature': True}
    ),
    install_requires=["numpy"],
)