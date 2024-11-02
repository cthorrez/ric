import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "src/ric.h":
    ctypedef struct _Dataset "Dataset":
        int (*matchups)[2]
        int* time_steps
        double* outcomes
        int num_matchups
        int num_competitors
    
    ctypedef struct _ModelInputs "ModelInputs":
        _Dataset* dataset
        double** model_params
        double* hyper_params
        double* probs
    
    ctypedef void (*RatingSystem)(_ModelInputs)
    
    # Function declarations
    void _online_elo "online_elo" (_ModelInputs)
    void _online_glicko "online_glicko" (_ModelInputs)
    void _online_trueskill "online_trueskill" (_ModelInputs)
    void _compute_metrics "compute_metrics" (double[], double[], double[3], int)
    double _evaluate "evaluate" (RatingSystem, _ModelInputs, double[3])

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
    cdef _Dataset dataset = _Dataset(<int (*)[2]>matchups.data, NULL, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double* ratings_ptr = &ratings[0,0]
    cdef _ModelInputs inputs = _ModelInputs(&dataset, &ratings_ptr, &hyper_params[0], &probs[0])
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
    cdef _Dataset dataset = _Dataset(<int (*)[2]>matchups.data, <int*>time_steps.data, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double** params_ptr = <double**>malloc(2 * sizeof(double*))
    params_ptr[0] = <double*>ratings.data
    params_ptr[1] = <double*>rd2s.data
    cdef _ModelInputs inputs = _ModelInputs(&dataset, params_ptr, &hyper_params[0], &probs[0])
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
    cdef np.ndarray[double, ndim=1] mus = np.full(num_competitors, initial_mu, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] sigma2s = np.full(num_competitors, initial_sigma*initial_sigma, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([beta, tau, epsilon], dtype=np.float64)
    cdef _Dataset dataset = _Dataset(<int (*)[2]>matchups.data, NULL, <double*>outcomes.data, num_matchups, num_competitors)
    cdef double** params_ptr = <double**>malloc(2 * sizeof(double*))
    params_ptr[0] = <double*>mus.data
    params_ptr[1] = <double*>sigma2s.data
    cdef _ModelInputs inputs = _ModelInputs(&dataset, params_ptr, &hyper_params[0], &probs[0])
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


# Create Python-accessible classes
cdef class Dataset:
    cdef _Dataset _c_dataset

    def __cinit__(self, np.ndarray[int, ndim=2] matchups, np.ndarray[double, ndim=1] outcomes, 
                  np.ndarray[int, ndim=1] time_steps=None):
        self._c_dataset.matchups = <int (*)[2]>matchups.data
        self._c_dataset.outcomes = <double*>outcomes.data
        self._c_dataset.num_matchups = matchups.shape[0]
        self._c_dataset.num_competitors = np.max(matchups) + 1
        if time_steps is not None:
            self._c_dataset.time_steps = <int*>time_steps.data
        else:
            self._c_dataset.time_steps = NULL

cdef class ModelInputs:
    cdef _ModelInputs _c_inputs
    cdef object _keep_alive  # Keep Python objects alive
    cdef double** _params_ptr

    def __cinit__(self, Dataset dataset, np.ndarray[double, ndim=2] model_params, 
                  np.ndarray[double, ndim=1] hyper_params, np.ndarray[double, ndim=1] probs=None):
        self._c_inputs.dataset = &dataset._c_dataset
        
        # Allocate and set model_params pointer
        self._params_ptr = <double**>malloc(2 * sizeof(double*))
        self._params_ptr[0] = <double*>&model_params[0,0]
        if model_params.shape[1] > 1:
            self._params_ptr[1] = <double*>&model_params[0,1]
            
        self._c_inputs.model_params = self._params_ptr
        self._c_inputs.hyper_params = <double*>hyper_params.data
        
        if probs is None:
            probs = np.zeros(dataset._c_dataset.num_matchups, dtype=np.float64)
        self._c_inputs.probs = <double*>probs.data
        
        # Keep references to prevent garbage collection
        self._keep_alive = (dataset, model_params, hyper_params, probs)

    def __dealloc__(self):
        if self._params_ptr != NULL:
            free(self._params_ptr)

cpdef evaluate(str system_name, ModelInputs inputs):
    cdef RatingSystem rating_system
    cdef np.ndarray[double, ndim=1] metrics = np.zeros(3, dtype=np.float64)
    
    if system_name == "elo":
        rating_system = _online_elo
    elif system_name == "glicko":
        rating_system = _online_glicko
    elif system_name == "trueskill":
        rating_system = _online_trueskill
    else:
        raise ValueError(f"Unknown rating system: {system_name}")
    
    return _evaluate(rating_system, inputs._c_inputs, &metrics[0])
