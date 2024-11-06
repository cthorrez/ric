#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"


double* construct_elo_ratings(
    ModelInputs model_inputs
)
{
    double* ratings = malloc(model_inputs.num_competitors * sizeof(double));
    for (int i=0; i<model_inputs.num_competitors; i++){
        ratings[i] = model_inputs.hyper_params[0];
    }
    return ratings;
}


ModelOutputs online_elo(Dataset dataset, ModelInputs model_inputs)
{
    // model_inputs.hyper_params = [initial_rating, k, scale, base]
    const int (*matchups)[2] = dataset.matchups;
    double* outcomes = dataset.outcomes;
    const int num_matchups = dataset.num_matchups;
    double* ratings = construct_elo_ratings(model_inputs);
    double* probs = malloc(dataset.num_matchups * sizeof(double));
    double k = model_inputs.hyper_params[1];
    double scale = model_inputs.hyper_params[2];
    double base = model_inputs.hyper_params[3];
    const double alpha = log(base) / scale;
    int idx_a, idx_b;
    double logit, prob, update;
    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        logit = alpha * (ratings[idx_b] - ratings[idx_a]);
        prob = 1.0 / (1.0 + exp(logit));
        probs[i] = prob;
        update = k * (outcomes[i] - prob);
        ratings[idx_a] += update;
        ratings[idx_b] -= update;
    }
    ModelOutputs model_outputs = {probs, ratings};
    return model_outputs;
}