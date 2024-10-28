#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "ric.h"

void online_trueskill(
    int matchups[][2],
    double outcomes[],
    double mean[],
    double var[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double beta,
    double tau,
    double epsilon
)
{
    double beta2 = beta * beta;
    

    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        mean[idx_a] += 1.0;
        mean[idx_b] += 1.0;
    }
}