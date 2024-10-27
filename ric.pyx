import numpy as np
cimport numpy as np

cdef extern from "src/ric.h":
    void _online_elo "online_elo" (int[][2], double[], int, int, double, double, double, double[], double[])
    void _online_glicko "online_glicko" (int[][2], double[], int[], int, int, double, double, double, double, double[], double[], double[])


def online_elo(np.ndarray[int, ndim=2] matchups,
                 np.ndarray[double, ndim=1] outcomes,
                 int num_matchups,
                 int num_competitors,
                 double initial_rating,
                 double k,
                 double scale,
                 double base,
                ):
    cdef np.ndarray[double, ndim=1] ratings = np.full(num_competitors, fill_value=initial_rating, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _online_elo(<int (*)[2]>matchups.data, &outcomes[0], num_matchups, num_competitors, k, scale, base, &ratings[0], &probs[0])
    return ratings, probs

def online_glicko(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[double, ndim=1] outcomes,
    np.ndarray[int, ndim=1] time_steps,
    int num_matchups,
    int num_competitors,
    double initial_r,
    double initial_rd,
    double c,
    double scale,
    double base,
):
    cdef np.ndarray[double, ndim=1] rs = np.full(num_competitors, fill_value=initial_r, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] rd2s = np.full(num_competitors, fill_value=initial_rd*initial_rd, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _online_glicko(<int (*)[2]>matchups.data, &outcomes[0], &time_steps[0], num_matchups, num_competitors, initial_rd, c, scale, base, &rs[0], &rd2s[0], &probs[0])
    cdef np.ndarray[double, ndim=1] rds = np.sqrt(rd2s)
    return rs, rds, probs
