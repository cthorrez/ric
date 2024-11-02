import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "src/ric.h":
    ctypedef struct Dataset:
        int (*matchups)[2]
        int* time_steps
        double* outcomes
        int num_matchups
        int num_competitors
    
    ctypedef struct ModelInputs:
        Dataset* dataset
        double** model_params
        double* hyper_params
        double* probs
    
    ctypedef void (*RatingSystem)(ModelInputs model_inputs)
    
    # Function declarations
    void _online_elo "online_elo" (ModelInputs)
    void _online_glicko "online_glicko" (ModelInputs)
    void _online_trueskill "online_trueskill" (ModelInputs)
    void _compute_metrics "compute_metrics" (double[], double[], double[3], int)
    void _evaluate "evaluate" (RatingSystem, Dataset, double[3], int)

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
    # Initialize arrays
    cdef np.ndarray[double, ndim=2] ratings = np.full((num_competitors, 1), initial_rating, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([k, scale, base], dtype=np.float64)
    cdef Dataset dataset = Dataset(<int (*)[2]>matchups.data, NULL, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double* ratings_ptr = &ratings[0,0]
    cdef ModelInputs inputs = ModelInputs(&dataset, &ratings_ptr, &hyper_params[0], &probs[0])
    _online_elo(inputs)
    return ratings[:,0], probs

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
    cdef np.ndarray[double, ndim=1] ratings = np.full(num_competitors, initial_rating, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] rd2s = np.full(num_competitors, initial_rd * initial_rd, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([initial_rd, c, scale, base], dtype=np.float64)
    cdef Dataset dataset = Dataset(<int (*)[2]>matchups.data, <int*>time_steps.data, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double** params_ptr = <double**>malloc(2 * sizeof(double*))
    params_ptr[0] = <double*>ratings.data
    params_ptr[1] = <double*>rd2s.data
    cdef ModelInputs inputs = ModelInputs(&dataset, params_ptr, &hyper_params[0], &probs[0])
    _online_glicko(inputs)
    free(params_ptr)
    return ratings, np.sqrt(rd2s), probs


cpdef online_trueskill(
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
    """
    Adds two integers and returns the result.

    Parameters
    ----------
    a : int
        First integer to add.
    b : int
        Second integer to add.

    Returns
    -------
    int
        The sum of `a` and `b`.
    """
    cdef np.ndarray[double, ndim=1] mus = np.full(num_competitors, initial_mu, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] sigma2s = np.full(num_competitors, initial_sigma*initial_sigma, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([beta, tau, epsilon], dtype=np.float64)
    cdef Dataset dataset = Dataset(<int (*)[2]>matchups.data, NULL, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double** params_ptr = <double**>malloc(2 * sizeof(double*))
    params_ptr[0] = <double*>mus.data
    params_ptr[1] = <double*>sigma2s.data
    cdef ModelInputs inputs = ModelInputs(&dataset, params_ptr, &hyper_params[0], &probs[0])
    _online_trueskill(inputs)
    free(params_ptr)
    return mus, np.sqrt(sigma2s), probs

def compute_metrics(
    np.ndarray[double, ndim=1] probs,
    np.ndarray[double, ndim=1] outcomes,
):
    cdef np.ndarray[double, ndim=1] metrics = np.zeros(3, dtype=np.float64)
    _compute_metrics(&probs[0], &outcomes[0], &metrics[0], probs.shape[0])
    return metrics
