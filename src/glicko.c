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

void online_glicko(ModelInputs model_inputs)
/*
 * Updates Glicko ratings based on match outcomes
 * 
 * @param model_inputs Structure containing:
 *   dataset:
 *     - matchups[num_matchups][2]: pairs of competitor indices
 *     - time_steps[num_matchups]: time step for each match
 *     - outcomes[num_matchups]: match results
 *   model_params:
 *     - [0]: ratings[num_competitors]: rating for each competitor
 *     - [1]: rd2s[num_competitors]: rating deviation squared
 *   hyper_params: [max_rd, c, base, scale]
 *   probs[num_matchups]: output probabilities for each match
 */
{
    const int (*matchups)[2] = model_inputs.dataset->matchups;
    const int* time_steps = model_inputs.dataset->time_steps;
    const double* outcomes = model_inputs.dataset->outcomes;
    const int num_matchups = model_inputs.dataset->num_matchups;
    const int num_competitors = model_inputs.dataset->num_competitors;
    double* ratings = model_inputs.model_params[0];
    double* rd2s = model_inputs.model_params[1];
    double* probs = model_inputs.probs;
    const double* h = model_inputs.hyper_params;
    const double max_rd = h[0], c = h[1], base = h[2], scale = h[3];


    int idx_a, idx_b, last_played_a, last_played_b;
    double r_a, r_b, rd2_a, rd2_b, g_a, g_b, logit, prob_a, prob_b, d2_inv_a, d2_inv_b;
    const double q = log(base) / scale;
    const double q2 = q * q;
    const double c2 = c * c;
    const double three_q2_over_pi2 = (3.0 * q2) / M_PI;
    const double max_rd2 = max_rd * max_rd;
    int* last_played = (int*)calloc(num_competitors, sizeof(int));

    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        r_a = ratings[idx_a];
        r_b = ratings[idx_b];
        rd2_a = rd2s[idx_a];
        rd2_b = rd2s[idx_b];
        last_played_a = last_played[idx_a];
        last_played_b = last_played[idx_b];

        // increase rd for passage of time
        if (last_played_a != time_steps[i]) {
            rd2_a += (double) ((time_steps[i] - last_played_a) * c2);
            if (rd2_a > max_rd2) {
                rd2_a = max_rd2;
            }
        }
        if (last_played_b != time_steps[i]) {
            rd2_b += (double) ((time_steps[i] - last_played_b) * c2);
            if (rd2_b > max_rd2) {
                rd2_b = max_rd2;
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

        rd2s[idx_a] = 1.0 / ((1.0 / rd2_a) + d2_inv_a);
        rd2s[idx_b] = 1.0 / ((1.0 / rd2_b) + d2_inv_b);

        ratings[idx_a] += q * rd2s[idx_a] * g_b * (outcomes[i] - prob_a);
        ratings[idx_b] += q * rd2s[idx_b] * g_a * (1.0 - outcomes[i] - prob_b);

        last_played[idx_a] = time_steps[i];
        last_played[idx_b] = time_steps[i];
    }
    free(last_played);
}