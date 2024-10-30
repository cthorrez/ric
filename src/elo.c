#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

void online_elo(
    const int matchups[][2],
    const double outcomes[],
    double mean[],
    double probs[],
    const int num_matchups,
    const int num_competitors,
    const double k,
    const double scale,
    const double base
)
{
    const double alpha = log(base) / scale;
    int idx_a, idx_b;
    double logit, prob, update;
    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        logit = alpha * (mean[idx_b] - mean[idx_a]);
        prob = 1.0 / (1.0 + exp(logit));
        probs[i] = prob;
        update = k * (outcomes[i] - prob);
        mean[idx_a] += update;
        mean[idx_b] -= update;
    }
}