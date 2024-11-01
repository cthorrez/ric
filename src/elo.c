#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

void online_elo(ModelInputs model_inputs)
{
    printf("1\n");
    fflush(stdout);  // Force print
    
    if (model_inputs.dataset == NULL) {
        printf("dataset is NULL\n");
        fflush(stdout);
        return;
    }
    printf("2\n");
    fflush(stdout);
    
    if (model_inputs.dataset->matchups == NULL) {
        printf("matchups is NULL\n");
        fflush(stdout);
        return;
    }
    printf("3\n");
    fflush(stdout);
    
    // Then try to access the first values
    printf("first matchup: %d vs %d\n", 
           model_inputs.dataset->matchups[0][0],
           model_inputs.dataset->matchups[0][1]);
    fflush(stdout);

    const int (*matchups)[2] = model_inputs.dataset->matchups;
    double* outcomes = model_inputs.dataset->outcomes;
    const int num_matchups = model_inputs.dataset->num_matchups;
    double* ratings = model_inputs.model_params[0];
    double* probs = model_inputs.probs;
    double k = model_inputs.hyper_params[0];
    double base = model_inputs.hyper_params[1];
    double scale = model_inputs.hyper_params[2];
    const double alpha = log(base) / scale;
    int idx_a, idx_b;
    double logit, prob, update;
    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];

        // Try accessing each other field
        printf("accessing ratings pointer\n");
        fflush(stdout);
        double* ratings = model_inputs.model_params[0];
        printf("ratings pointer: %p\n", (void*)ratings);
        fflush(stdout);

        printf("accessing ratings values\n");
        fflush(stdout);
        printf("rating[%d]: %f\n", idx_a, ratings[idx_a]);
        printf("rating[%d]: %f\n", idx_b, ratings[idx_b]);
        fflush(stdout);

        printf("accessing other fields\n");
        fflush(stdout);


        logit = alpha * (ratings[idx_b] - ratings[idx_a]);
        prob = 1.0 / (1.0 + exp(logit));
        probs[i] = prob;
        update = k * (outcomes[i] - prob);
        ratings[idx_a] += update;
        ratings[idx_b] -= update;
    }
}