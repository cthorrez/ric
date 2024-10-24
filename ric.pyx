import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "src/ric.h":
    double* run_elo(int[][2], double[], int, double, double, double)

def calculate_elo(np.ndarray[int, ndim=2] matchups,
                 np.ndarray[double, ndim=1] outcomes,
                 int num_competitors,
                 double initial_rating=1500.0,
                 double scale=400.0,
                 double base=32.0):
    cdef double* ratings = run_elo(<int (*)[2]>matchups.data, &outcomes[0], num_competitors, initial_rating, scale, base)
    cdef np.ndarray[double, ndim=1] np_ratings = np.zeros(num_competitors, dtype=np.float64)
    for i in range(num_competitors):
        np_ratings[i] = ratings[i]
    free(ratings)
    return np_ratings