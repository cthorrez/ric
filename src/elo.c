#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

void online_elo(ModelInputs model_inputs)
{
    const int (*matchups)[2] = model_inputs.dataset->matchups;
    double* outcomes = model_inputs.dataset->outcomes;
    const int num_matchups = model_inputs.dataset->num_matchups;
    double* ratings = model_inputs.model_params[0];
    double* probs = model_inputs.probs;
    double k = model_inputs.hyper_params[0];
    double scale = model_inputs.hyper_params[1];
    double base = model_inputs.hyper_params[2];
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
}