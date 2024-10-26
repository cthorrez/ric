#include <stdlib.h>
#include <stdio.h>
#define _XOPEN_SOURCE
#define _USE_MATH_DEFINES
#include <math.h>
#include "ric.h"

void online_glicko(
    int matchups[][2],
    double outcomes[],
    int time_steps[],
    int num_matchups,
    int num_competitors,
    double initial_rd,
    double c,
    double scale,
    double base,
    double rs[],
    double rds[],
    double probs[]
)
{
    
    double q = log(base) / scale;
    double q2 = q * q;
    double three_q2_over_pi2 = (3.0 * q2) / M_PI;

    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        rs[idx_a] += M_PI;
        rs[idx_b] -= M_PI;
    }
}