import math
import numpy as np
from scipy.stats import norm
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "src/ric.h":
    ctypedef struct _Dataset "Dataset":
        int (*matchups)[2]
        int* time_steps
        double* outcomes
        int num_matchups
    
    ctypedef struct _ModelInputs "ModelInputs":
        double* hyper_params
        int num_competitors
    
    ctypedef struct _ModelOutputs "ModelOutputs":
        double* ratings
        double* probs

    ctypedef struct _SweepOutputs "SweepOutputs":
        double* best_metrics
        _ModelInputs best_inputs
    
    _ModelOutputs _online_elo "online_elo" (_Dataset, _ModelInputs)
    _ModelOutputs _online_glicko "online_glicko" (_Dataset, _ModelInputs)
    _ModelOutputs _online_trueskill "online_trueskill" (_Dataset, _ModelInputs)


    ctypedef _ModelOutputs (*RatingSystem)(_Dataset, _ModelInputs)
    void _compute_metrics "compute_metrics" (double[], double[], double[3], int)
    _SweepOutputs _sweep "sweep" (RatingSystem, _Dataset, _ModelInputs*, int, int)

    # multi dataset/params functions:
    _ModelOutputs* _multi_dataset_fit "multi_dataset_fit" (RatingSystem, _Dataset*, _ModelInputs, int, int)
    _ModelOutputs* _multi_params_fit "multi_params_fit" (RatingSystem, _Dataset, _ModelInputs*, int, int)


def online_elo(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    double initial_rating=1500.0,
    double k=32.0,
    double scale=400.0,
    double base=10.0,
):
    cdef int num_matchups = matchups.shape[0]
    cdef _Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    dataset.outcomes = <double*>outcomes.data
    dataset.time_steps = NULL
    dataset.num_matchups = num_matchups

    cdef np.ndarray[double, ndim=1] hyper_params = np.array([initial_rating, k, scale, base], dtype=np.float64)
    cdef _ModelInputs model_inputs
    model_inputs.hyper_params = <double*>hyper_params.data
    model_inputs.num_competitors = num_competitors

    # Call C function
    cdef _ModelOutputs outputs = _online_elo(dataset, model_inputs)
    
    # Convert output pointers to numpy arrays
    cdef np.ndarray[double, ndim=1] ratings = np.zeros(num_competitors, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(num_matchups, dtype=np.float64)
    
    cdef np.npy_intp ratings_dim = num_competitors
    cdef np.npy_intp probs_dim = num_matchups

    ratings = np.PyArray_SimpleNewFromData(1, &ratings_dim, np.NPY_DOUBLE, outputs.ratings)
    probs = np.PyArray_SimpleNewFromData(1, &probs_dim, np.NPY_DOUBLE, outputs.probs)
        
    return ratings, probs

def online_glicko(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[int, ndim=1] time_steps,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    double initial_rating=1500.0,
    double initial_rd=350.0,
    double c=63.2,
    double scale=400.0,
    double base=10.0,
):
    cdef int num_matchups = matchups.shape[0]
    
    # Create dataset struct
    cdef _Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    dataset.time_steps = <int*>time_steps.data
    dataset.outcomes = <double*>outcomes.data
    dataset.num_matchups = num_matchups
    
    # Create model inputs struct
    cdef np.ndarray[double, ndim=1] hyper_params = np.array(
        [initial_rating, initial_rd, c, scale, base], 
        dtype=np.float64
    )
    cdef _ModelInputs model_inputs
    model_inputs.hyper_params = <double*>hyper_params.data
    model_inputs.num_competitors = num_competitors
    
    cdef _ModelOutputs outputs = _online_glicko(dataset, model_inputs)
    
    cdef np.npy_intp ratings_dim = num_competitors
    cdef np.npy_intp probs_dim = num_matchups
    
    ratings = np.PyArray_SimpleNewFromData(1, &ratings_dim, np.NPY_DOUBLE, outputs.ratings)
    rd2s = np.PyArray_SimpleNewFromData(1, &ratings_dim, np.NPY_DOUBLE, outputs.ratings + num_competitors)
    probs = np.PyArray_SimpleNewFromData(1, &probs_dim, np.NPY_DOUBLE, outputs.probs)
    
    return ratings, np.sqrt(rd2s), probs


def online_trueskill(
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    double initial_mu=25.0,
    double initial_sigma=8.33,
    double beta=4.166,
    double tau=0.0833,
    double epsilon=0.0001,
):
    cdef int num_matchups = matchups.shape[0]
    
    # Create dataset struct
    cdef _Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    dataset.outcomes = <double*>outcomes.data
    dataset.time_steps = NULL
    dataset.num_matchups = num_matchups
    
    # Create model inputs struct
    cdef np.ndarray[double, ndim=1] hyper_params = np.array([initial_mu, initial_sigma, beta, tau, epsilon], dtype=np.float64)
    cdef _ModelInputs model_inputs
    model_inputs.hyper_params = <double*>hyper_params.data
    model_inputs.num_competitors = num_competitors
    
    cdef _ModelOutputs outputs = _online_trueskill(dataset, model_inputs)
    
    cdef np.npy_intp ratings_dim = num_competitors
    cdef np.npy_intp probs_dim = num_matchups
    
    mus = np.PyArray_SimpleNewFromData(1, &ratings_dim, np.NPY_DOUBLE, outputs.ratings)
    sigma2s = np.PyArray_SimpleNewFromData(1, &ratings_dim, np.NPY_DOUBLE, outputs.ratings + num_competitors)
    probs = np.PyArray_SimpleNewFromData(1, &probs_dim, np.NPY_DOUBLE, outputs.probs)
    
    return mus, np.sqrt(sigma2s), probs

def compute_metrics(
    np.ndarray[double, ndim=1] probs,
    np.ndarray[double, ndim=1] outcomes,
):
    cdef np.ndarray[double, ndim=1] metrics = np.zeros(3, dtype=np.float64)
    _compute_metrics(&probs[0], &outcomes[0], &metrics[0], probs.shape[0])
    return metrics



def sweep(
    str system_name,
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[int, ndim=1] time_steps,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    np.ndarray[double, ndim=2] param_sets,  # Each row is a set of hyperparameters
    int num_threads=24,
):
    cdef RatingSystem rating_system
    if system_name == "elo":
        rating_system = _online_elo
    elif system_name == "glicko":
        rating_system = _online_glicko
    elif system_name == "trueskill":
        rating_system = _online_trueskill
    else:
        raise ValueError(f"Unknown rating system: {system_name}")
    
    # Create dataset struct
    cdef _Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    dataset.time_steps = <int*>time_steps.data if time_steps is not None else NULL
    dataset.outcomes = <double*>outcomes.data
    dataset.num_matchups = matchups.shape[0]

    # Create array of ModelInputs for each parameter combination
    cdef int num_sweep_inputs = param_sets.shape[0]
    cdef _ModelInputs* sweep_inputs = <_ModelInputs*>malloc(num_sweep_inputs * sizeof(_ModelInputs))
    
    # Setup each ModelInputs struct
    for i in range(num_sweep_inputs):
        sweep_inputs[i].num_competitors = num_competitors
        sweep_inputs[i].hyper_params = &param_sets[i,0]
    
    # Call C sweep function
    cdef _SweepOutputs outputs = _sweep(
        rating_system,
        dataset,
        sweep_inputs,
        num_sweep_inputs,
        num_threads
    )
    
    # Extract best parameters
    cdef int num_params = param_sets.shape[1]
    cdef np.ndarray[double, ndim=1] best_params = np.zeros(num_params, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] best_metrics = np.zeros(3, dtype=np.float64)  # [log_loss, brier, accuracy]
    for i in range(num_params):
        best_params[i] = outputs.best_inputs.hyper_params[i]
    for i in range(3):
        best_metrics[i] = outputs.best_metrics[i]
    
    free(sweep_inputs)
    
    return best_metrics, best_params


# Create Python-accessible classes
cdef class Dataset:
    cdef _Dataset _c_dataset

    def __cinit__(self, np.ndarray[int, ndim=2] matchups, np.ndarray[double, ndim=1] outcomes, 
                  np.ndarray[int, ndim=1] time_steps=None):
        self._c_dataset.matchups = <int (*)[2]>matchups.data
        self._c_dataset.outcomes = <double*>outcomes.data
        self._c_dataset.num_matchups = matchups.shape[0]
        if time_steps is not None:
            self._c_dataset.time_steps = <int*>time_steps.data
        else:
            self._c_dataset.time_steps = NULL

cdef class ModelInputs:
    cdef _ModelInputs _c_inputs
    cdef object _keep_alive  # Keep Python objects alive

    def __cinit__(self, np.ndarray[double, ndim=1] hyper_params, int num_competitors):
        self._c_inputs.hyper_params = <double*>hyper_params.data
        self._c_inputs.num_competitors = num_competitors
        self._keep_alive = hyper_params

cdef class ModelOutputs:
    cdef _ModelOutputs _c_outputs
    cdef object _keep_alive

    def __cinit__(self, np.ndarray[double, ndim=1] ratings, np.ndarray[double, ndim=1] probs):
        self._c_outputs.ratings = <double*>ratings.data
        self._c_outputs.probs = <double*>probs.data
        self._keep_alive = (ratings, probs)


def sample_fit(
    str system_name,
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[int, ndim=1] time_steps,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    np.ndarray[double, ndim=1] params,
    int num_samples,
    bint replace=True,
    int num_threads=10,
    int batch_size=10,
    int seed=0,
):
    cdef RatingSystem rating_system
    rating_dim = 1
    if system_name == "elo":
        rating_system = _online_elo
    elif system_name == "glicko":
        rating_system = _online_glicko
        rating_dim = 2
    elif system_name == "trueskill":
        rating_system = _online_trueskill
        rating_dim = 2
    else:
        raise ValueError(f"Unknown rating system: {system_name}")

    cdef np.npy_intp num_rating_params = num_competitors * rating_dim
    rng = np.random.default_rng(seed)

    cdef np.ndarray[double, ndim=3] all_ratings = np.zeros((num_samples, num_competitors, rating_dim), dtype=np.float64)
    model_inputs = ModelInputs(params, num_competitors)
    num_batches = num_samples // batch_size

    cdef int start = 0
    cdef int effective_batch_size
    cdef _Dataset* datasets
    array_refs = []

    cdef np.ndarray[int, ndim=2] batch_matchups
    cdef np.ndarray[double, ndim=1] batch_outcomes
    cdef np.ndarray[int, ndim=1] batch_timesteps

    while start < num_samples:
        effective_batch_size = min(batch_size, num_samples - start)
        datasets = <_Dataset*>malloc(effective_batch_size * sizeof(_Dataset))
        if replace:
            idxs = rng.integers(0, matchups.shape[0], size=(effective_batch_size, matchups.shape[0]))
        else:
            idxs = rng.permutation(np.tile(np.arange(matchups.shape[0]), (effective_batch_size, 1)), axis=1)
        array_refs.clear()

        for j in range(effective_batch_size):
            batch_matchups = matchups[idxs[j]]
            batch_outcomes = outcomes[idxs[j]]
            array_refs.append((batch_matchups, batch_outcomes))
            datasets[j].matchups = <int (*)[2]>&batch_matchups[0, 0]
            datasets[j].outcomes = <double*>&batch_outcomes[0]
            datasets[j].num_matchups = batch_matchups.shape[0]
            if time_steps is not None:
                batch_timesteps = time_steps[idxs[j]]
                array_refs.append(batch_timesteps)
                datasets[j].time_steps = <int*>&batch_timesteps[0]
            else:
                datasets[j].time_steps = NULL

        outputs = _multi_dataset_fit(rating_system, datasets, model_inputs._c_inputs, effective_batch_size, num_threads)
        for j in range(effective_batch_size):
            ratings = np.PyArray_SimpleNewFromData(1, &num_rating_params, np.NPY_DOUBLE, outputs[j].ratings)
            ratings = ratings.reshape((num_competitors, rating_dim), order='F')
            all_ratings[start + j] = ratings
            free(outputs[j].ratings)
            free(outputs[j].probs)

        free(datasets)
        free(outputs)
        start += effective_batch_size

    return all_ratings


def sweep_batch_eval(
    str system_name,
    np.ndarray[int, ndim=2] matchups,
    np.ndarray[int, ndim=1] time_steps,
    np.ndarray[double, ndim=1] outcomes,
    int num_competitors,
    np.ndarray[double, ndim=2] param_sets,
    int num_threads=10,
    int batch_size=10,
):
    cdef RatingSystem rating_system
    cdef int rating_dim = 1
    if system_name == "elo":
        rating_system = _online_elo
    elif system_name == "glicko":
        rating_system = _online_glicko
        rating_dim = 2
    elif system_name == "trueskill":
        rating_system = _online_trueskill
        rating_dim = 2
    else:
        raise ValueError(f"Unknown rating system: {system_name}")

    cdef np.npy_intp num_rating_params = num_competitors * rating_dim
    cdef int num_param_sets = param_sets.shape[0]
    
    cdef _Dataset dataset
    dataset.matchups = <int (*)[2]>matchups.data
    dataset.outcomes = <double*>outcomes.data
    dataset.num_matchups = matchups.shape[0]
    dataset.time_steps = <int*>time_steps.data if time_steps is not None else NULL

    cdef int start = 0
    cdef int effective_batch_size
    cdef _ModelInputs* model_inputs = NULL
    cdef _ModelOutputs* outputs = NULL
    cdef int i, j
    cdef double* params_ptr = NULL
    cdef np.ndarray ratings_array

    best_ratings = np.zeros((num_competitors, rating_dim), dtype=np.float64)
    best_params = np.zeros(param_sets.shape[1], dtype=np.float64)
    best_log_loss = 1e9
    
    while start < num_param_sets:
        effective_batch_size = min(batch_size, num_param_sets - start)
        
        model_inputs = <_ModelInputs*>malloc(effective_batch_size * sizeof(_ModelInputs))
        if model_inputs == NULL:
            return None, None
            
        for i in range(effective_batch_size):
            model_inputs[i].hyper_params = NULL
            model_inputs[i].num_competitors = num_competitors
            
            params_ptr = <double*>malloc(param_sets.shape[1] * sizeof(double))
            if params_ptr == NULL:
                for j in range(i):
                    free(model_inputs[j].hyper_params)
                free(model_inputs)
                return None, None
                
            model_inputs[i].hyper_params = params_ptr
            for j in range(param_sets.shape[1]):
                params_ptr[j] = param_sets[start + i, j]
        
        outputs = _multi_params_fit(rating_system, dataset, model_inputs, 
                                  effective_batch_size, num_threads)
        
        for i in range(effective_batch_size):
            if outputs[i].ratings != NULL:
                ratings_array = np.PyArray_SimpleNewFromData(1, &num_rating_params, 
                                                          np.NPY_DOUBLE, outputs[i].ratings)
                ratings = ratings_array.reshape((num_competitors, rating_dim), order='F')

                if system_name == 'elo':
                    probs = predict_elo_batch(ratings[:,0], matchups, 
                                            scale=param_sets[start + i, 2], 
                                            base=param_sets[start + i, 3])
                elif system_name == 'glicko':
                    probs = predict_glicko_batch(ratings, matchups, 
                                               scale=param_sets[start + i, 3], 
                                               base=param_sets[start + i, 4])
                elif system_name == 'trueskill':
                    probs = predict_trueskill_batch(ratings, matchups, 
                                                  beta=param_sets[start + i, 2])

                metrics = compute_metrics(probs, outcomes)
                if metrics[1] < best_log_loss:
                    best_log_loss = metrics[1]
                    best_params = param_sets[start + i].copy()
                    best_ratings = ratings.copy()

            # Clean up resources for this iteration
            if outputs[i].ratings != NULL:
                free(outputs[i].ratings)
            if outputs[i].probs != NULL:
                free(outputs[i].probs)
            free(model_inputs[i].hyper_params)
        
        free(model_inputs)
        free(outputs)
        start += effective_batch_size

    return best_ratings, best_params



def predict_elo_batch(ratings, pairs, scale=400.0, base=10.0):
    alpha = np.log(base) / scale
    rating_diffs = ratings[pairs[:,1]] - ratings[pairs[:,0]]
    probs = 1.0 / (1.0 + np.exp(alpha * rating_diffs))
    return probs

def predict_glicko_batch(ratings, pairs, scale=400.0, base=10.0):
    q = np.log(base) / scale
    mus = ratings[:,0]
    rd2s = ratings[:,1]
    c_rd2s = rd2s[pairs[:,0]] + rd2s[pairs[:,1]]
    gs = 1.0 / np.sqrt(1.0 + (3*q*q*c_rd2s/(math.pi * math.pi)))
    mu_diffs = mus[pairs[:,1]] - mus[pairs[:,0]]
    probs = 1.0 / (1.0 + np.exp(q * gs * mu_diffs))
    return probs

def predict_trueskill_batch(ratings, pairs, beta=4.166):
    mu = ratings[:,0]
    sigma2 = ratings[:,1]
    mu_diffs = mu[pairs[:,0]] - mu[pairs[:,1]]
    c = np.sqrt(2.0 * (beta * beta) + sigma2[pairs[:,0]] + sigma2[pairs[:,1]])
    probs = norm.cdf(mu_diffs / c)
    return probs    