#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "ric.h"



double g(const double rd2, const double three_q2_over_pi2){
    return 1.0 / sqrt(1.0 + (rd2 * three_q2_over_pi2));
}

double calc_prob(const double logit, const double g_opp) {
    return 1.0 / (1.0 + exp(logit * g_opp));
}

void online_glicko(
    const int matchups[][2],
    const int time_steps[],
    const double outcomes[],
    double mean[],
    double var[],
    double probs[],
    const int num_matchups,
    const int num_competitors,
    const double max_rd,
    const double c,
    const double scale,
    const double base
)
{
    int idx_a, idx_b, last_played_a, last_played_b;
    double r_a, r_b, rd2_a, rd2_b, g_a, g_b, logit, prob_a, prob_b, d2_inv_a, d2_inv_b;
    const double q = log(base) / scale;
    const double q2 = q * q;
    const double c2 = c * c;
    const double three_q2_over_pi2 = (3.0 * q2) / M_PI;
    const double max_var = max_rd * max_rd;
    int* last_played = (int*)calloc(num_competitors, sizeof(int));

    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        r_a = mean[idx_a];
        r_b = mean[idx_b];
        rd2_a = var[idx_a];
        rd2_b = var[idx_b];
        last_played_a = last_played[idx_a];
        last_played_b = last_played[idx_b];

        // increase rd for passage of time
        if (last_played_a != time_steps[i]) {
            rd2_a += (double) ((time_steps[i] - last_played_a) * c2);
            if (rd2_a > max_var) {
                rd2_a = max_var;
            }
        }
        if (last_played_b != time_steps[i]) {
            rd2_b += (double) ((time_steps[i] - last_played_b) * c2);
            if (rd2_b > max_var) {
                rd2_b = max_var;
            }
        }
        
        g_a = g(rd2_a, three_q2_over_pi2);
        g_b = g(rd2_b, three_q2_over_pi2);

        logit = q * (r_b - r_a);
        prob_a = calc_prob(logit, g_b);
        prob_b = calc_prob(-logit, g_a);
        probs[i] = (prob_a + 1.0 - prob_b) / 2.0;

        d2_inv_a = q2 * g_b * g_b * prob_a * (1.0 - prob_a);
        d2_inv_b = q2 * g_a * g_a * prob_b * (1.0 - prob_b);

        var[idx_a] = 1.0 / ((1.0 / rd2_a) + d2_inv_a);
        var[idx_b] = 1.0 / ((1.0 / rd2_b) + d2_inv_b);

        mean[idx_a] += q * var[idx_a] * g_b * (outcomes[i] - prob_a);
        mean[idx_b] += q * var[idx_b] * g_a * (1.0 - outcomes[i] - prob_b);

        last_played[idx_a] = time_steps[i];
        last_played[idx_b] = time_steps[i];
    }
    free(last_played);
}