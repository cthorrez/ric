import numpy as np
cimport numpy as np

cdef extern from "src/ric.h":
    void _online_elo "online_elo" (int[][2], double[], double[], double[], int, int, double, double, double)
    void _online_glicko "online_glicko" (int[][2], int[], double[], double[], double[], double[],  int, int, double, double, double, double)
    void _online_trueskill "online_trueskill" (int[][2], double[], double[], double[], double[], int, int, double, double, double)

def online_elo(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[double, ndim=1] outcomes,
    int num_matchups,
    int num_competitors,
    double initial_rating=1500.0,
    double k=32.0,
    double scale=400.0,
    double base=10.0,
):
    cdef np.ndarray[double, ndim=1] mean = np.full(num_competitors, fill_value=initial_rating, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _online_elo(<int (*)[2]>matchups.data, &outcomes[0], &mean[0], &probs[0], num_matchups, num_competitors, k, scale, base)
    return mean, probs

def online_glicko(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[int, ndim=1] time_steps,
    np.ndarray[double, ndim=1] outcomes,
    int num_matchups,
    int num_competitors,
    double initial_rating=1500.0,
    double initial_rd=350.0,
    double c=63.2,
    double scale=400.0,
    double base=10.0,
):
    cdef np.ndarray[double, ndim=1] mean = np.full(num_competitors, fill_value=initial_rating, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] var = np.full(num_competitors, fill_value=initial_rd*initial_rd, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _online_glicko(<int (*)[2]>matchups.data, &time_steps[0], &outcomes[0], &mean[0], &var[0], &probs[0], num_matchups, num_competitors, initial_rd, c, scale, base)
    var = np.sqrt(var)
    return mean, var, probs

def online_trueskill(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[double, ndim=1] outcomes,
    int num_matchups,
    int num_competitors,
    double initial_mu=25.0,
    double initial_sigma=8.33,
    double beta=4.166,
    double tau=0.0833,
    double epsilon=0.0001
):
    cdef np.ndarray[double, ndim=1] mean = np.full(num_competitors, fill_value=initial_mu, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] var = np.full(num_competitors, fill_value=initial_sigma*initial_sigma, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    _online_trueskill(
        <int (*)[2]>matchups.data, 
        &outcomes[0], 
        &mean[0],
        &var[0], 
        &probs[0],
        num_matchups, 
        num_competitors, 
        beta, 
        tau, 
        epsilon
    )
    
    var = np.sqrt(var)
    return mean, var, probs
