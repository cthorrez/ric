#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "ric.h"



double g(double rd2, double three_q2_over_pi2){
    return 1.0 / sqrt(1.0 + (rd2 * three_q2_over_pi2));
}

double calc_prob(double logit, double g_opp) {
    return 1.0 / (1.0 + exp(logit * g_opp));
}

void online_glicko(
    int matchups[][2],
    int time_steps[],
    double outcomes[],
    double mean[],
    double var[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double max_rd,
    double c,
    double scale,
    double base
)
{
    double q = log(base) / scale;
    double q2 = q * q;
    double c2 = c * c;
    double three_q2_over_pi2 = (3.0 * q2) / M_PI;
    double max_var = max_rd * max_rd;
    int* last_played = (int*)calloc(num_competitors, sizeof(int));

    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        double r_a = mean[idx_a];
        double r_b = mean[idx_b];
        double rd2_a = var[idx_a];
        double rd2_b = var[idx_b];
        int last_played_a = last_played[idx_a];
        int last_played_b = last_played[idx_b];

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
        
        double g_a = g(rd2_a, three_q2_over_pi2);
        double g_b = g(rd2_b, three_q2_over_pi2);
        double logit = q * (r_b - r_a);
        double prob_a = calc_prob(logit, g_b);
        double prob_b = calc_prob(-logit, g_a);
        probs[i] = (prob_a + 1.0 - prob_b) / 2.0;

        double d2_inv_a = q2 * g_b * g_b * prob_a * (1.0 - prob_a);
        double d2_inv_b = q2 * g_a * g_a * prob_b * (1.0 - prob_b);

        double new_rd2_a = 1.0 / ((1.0 / rd2_a) + d2_inv_a);
        double new_rd2_b = 1.0 / ((1.0 / rd2_b) + d2_inv_b);

        mean[idx_a] += q * new_rd2_a * g_b * (outcomes[i] - prob_a);
        mean[idx_b] += q * new_rd2_b * g_a * (1.0 - outcomes[i] - prob_b);

        var[idx_a] = new_rd2_a;
        var[idx_b] = new_rd2_b;

        last_played[idx_a] = time_steps[i];
        last_played[idx_b] = time_steps[i];
    }

    free(last_played);
}