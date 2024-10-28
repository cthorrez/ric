#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

void online_elo(
    int matchups[][2],
    double outcomes[],
    double mean[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double k,
    double scale,
    double base
)
{
    double alpha = log(base) / scale;
    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        double logit = alpha * (mean[idx_b] - mean[idx_a]);
        double prob = 1.0 / (1.0 + exp(logit));
        probs[i] = prob;
        double update = k * (outcomes[i] - prob);
        mean[idx_a] += update;
        mean[idx_b] -= update;
    }
}