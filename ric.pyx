import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void* PyDataMem_NEW(size_t size)
    int _import_array()

np.import_array()  # Initialize NumPy C-API

cdef extern from "src/ric.h":
    double* _run_elo "run_elo" (int[][2], double[], int, int, double, double, double)

def run_elo(np.ndarray[int, ndim=2] matchups,
                 np.ndarray[double, ndim=1] outcomes,
                 int num_matchups,
                 int num_competitors,
                 double initial_rating=1500.0,
                 double scale=400.0,
                 double base=32.0):
    # Allocate memory using NumPy's allocator
    cdef double* ratings = <double*>PyDataMem_NEW(num_competitors * sizeof(double))
    # Call C function with pre-allocated memory
    ratings = _run_elo(<int (*)[2]>matchups.data, &outcomes[0], num_matchups, num_competitors, initial_rating, scale, base)
    cdef np.npy_intp dims = num_competitors
    cdef np.ndarray[double, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, ratings)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr