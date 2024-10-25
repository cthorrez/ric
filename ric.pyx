import numpy as np
cimport numpy as np

cdef extern from "src/ric.h":
    void _run_elo "run_elo" (int[][2], double[], int, int, double, double, double, double, double[], double[])

def run_elo(np.ndarray[int, ndim=2] matchups,
                 np.ndarray[double, ndim=1] outcomes,
                 int num_matchups,
                 int num_competitors,
                 double initial_rating,
                 double k,
                 double scale,
                 double base,
                ):
    cdef np.ndarray[double, ndim=1] ratings = np.zeros(num_competitors, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _run_elo(<int (*)[2]>matchups.data, &outcomes[0], num_matchups, num_competitors, initial_rating, k, scale, base, &ratings[0], &probs[0])
    return ratings, probs