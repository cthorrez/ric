#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

void run_elo(
    int matchups[][2],
    double outcomes[],
    int num_matchups,
    int num_competitors,
    double initial_rating,
    double k,
    double scale,
    double base,
    double ratings[],
    double probs[]
)
{
    double alpha = log(base) / scale;
    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        double logit = alpha * (ratings[idx_b] - ratings[idx_a]);
        double prob = 1.0 / (1.0 + exp(logit));
        probs[i] = prob;
        double update = k * (outcomes[i] - prob);
        ratings[idx_a] += update;
        ratings[idx_b] -= update;
    }
}