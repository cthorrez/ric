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
    
    # Function declarations
    void _online_elo "online_elo" (ModelInputs)
    void _online_glicko "online_glicko" (ModelInputs)
    void _online_trueskill "online_trueskill" (ModelInputs)
    void _compute_metrics "compute_metrics" (double[], double[], double[3], int)

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
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([k, base, scale], dtype=np.float64)
    print('arrays')
    cdef Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    print('matchups')
    dataset.outcomes = <double*>outcomes.data
    print('outcomes')
    dataset.time_steps = NULL
    dataset.num_matchups = num_matchups
    dataset.num_competitors = num_competitors
    print('counts')
    
    cdef double** ratings_ptr = <double**>malloc(sizeof(double*))
    ratings_ptr[0] = &ratings[0,0]  # Point to first (and only) column

    cdef ModelInputs inputs
    inputs.dataset = &dataset
    print('dataset')
    inputs.model_params = ratings_ptr
    print('model params')
    inputs.hyper_params = &hyper_params[0]
    print('hparams')
    inputs.probs = &probs[0]
    print('probs')

    print('inputs')
    _online_elo(inputs)
    print('done')
    return ratings[:,0], probs

# def online_glicko(
#     np.ndarray[int, ndim=2] matchups,
#     np.ndarray[int, ndim=1] time_steps,
#     np.ndarray[double, ndim=1] outcomes,
#     int num_matchups,
#     int num_competitors,
#     double initial_rating=1500.0,
#     double initial_rd=350.0,
#     double c=63.2,
#     double scale=400.0,
#     double base=10.0,
# ):
#     cdef np.ndarray[double, ndim=1] mean = np.full(num_competitors, fill_value=initial_rating, dtype=np.float64)
#     cdef np.ndarray[double, ndim=1] var = np.full(num_competitors, fill_value=initial_rd*initial_rd, dtype=np.float64)
#     cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
#     _online_glicko(<int (*)[2]>matchups.data, &time_steps[0], &outcomes[0], &mean[0], &var[0], &probs[0], num_matchups, num_competitors, initial_rd, c, scale, base)
#     var = np.sqrt(var)
#     return mean, var, probs

# def online_trueskill(
#     np.ndarray[int, ndim=2] matchups,
#     np.ndarray[double, ndim=1] outcomes,
#     int num_matchups,
#     int num_competitors,
#     double initial_mu=25.0,
#     double initial_sigma=8.33,
#     double beta=4.166,
#     double tau=0.0833,
#     double epsilon=0.0001
# ):
#     cdef np.ndarray[double, ndim=1] mean = np.full(num_competitors, fill_value=initial_mu, dtype=np.float64)
#     cdef np.ndarray[double, ndim=1] var = np.full(num_competitors, fill_value=initial_sigma*initial_sigma, dtype=np.float64)
#     cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
#     _online_trueskill(
#         <int (*)[2]>matchups.data, 
#         &outcomes[0], 
#         &mean[0],
#         &var[0], 
#         &probs[0],
#         num_matchups, 
#         num_competitors, 
#         beta, 
#         tau, 
#         epsilon
#     )
#     var = np.sqrt(var)
#     return mean, var, probs

def compute_metrics(
    np.ndarray[double, ndim=1] probs,
    np.ndarray[double, ndim=1] outcomes,
):
    cdef np.ndarray[double, ndim=1] metrics = np.zeros(3, dtype=np.float64)
    _compute_metrics(&probs[0], &outcomes[0], &metrics[0], probs.shape[0])
    return metrics
