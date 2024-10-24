import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "src/ric.h":
    double* _run_elo "run_elo" (int[][2], double[], int, int, double, double, double)

def run_elo(np.ndarray[int, ndim=2] matchups,
                 np.ndarray[double, ndim=1] outcomes,
                 int num_matchups,
                 int num_competitors,
                 double initial_rating=1500.0,
                 double scale=400.0,
                 double base=32.0):
    cdef double* ratings = _run_elo(<int (*)[2]>matchups.data, &outcomes[0], num_matchups, num_competitors, initial_rating, scale, base)
    return np.PyArray_SimpleNewFromData(1, [num_competitors], np.NPY_FLOAT64, <void*>ratings)