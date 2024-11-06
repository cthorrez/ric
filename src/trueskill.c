#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "ric.h"


#define M_1_SQRT_2PI 0.39894228040143267793994605993438
double norm_pdf(const double z){
    return M_1_SQRT_2PI * exp(-0.5 * z * z);
}

double norm_cdf(const double z){
    return 0.5 * (1.0 + erf(z * M_SQRT1_2));
}

void v_w_win(const double t, const double epsilon, double* v, double* w) {
    double z = t - epsilon;
    double pdf = norm_pdf(z);
    double cdf = norm_cdf(z);
    const double MIN_CDF = 2.222758749e-162;  
    if (cdf > MIN_CDF) {
        *v = pdf / cdf;
    } else {
        *v = -z;
    }
    *w = *v * (*v + z);
}

void v_w_draw(const double t, const double epsilon, double* v, double* w){
    double e_p_t = epsilon + t;
    double e_m_t = epsilon - t;
    double denom = norm_cdf(e_m_t) + norm_cdf(e_p_t) - 1.0; // this relies on cdf(-epsilon - t) = 1 - cdf(epsilon + t)
    double pdf_m = norm_pdf(e_m_t);
    double pdf_p = norm_pdf(e_p_t); // this relies on pdf(epsilon + t) = pdf(-epsilon - t)
    *v = (pdf_p - pdf_m) / denom;
    *w = ((e_m_t * pdf_m) + (e_p_t * pdf_p)) / denom;
}

void online_trueskill(Dataset dataset, ModelInputs model_inputs)
/*
 * Updates TrueSkill ratings based on match outcomes
 * 
 * @param model_inputs:
 *   dataset:
 *     - matchups[num_matchups][2]: pairs of competitor indices
 *     - outcomes[num_matchups]: match results (0, 0.5, or 1)
 *   model_params:
 *     - [0]: mus[num_competitors]: skill mean for each competitor
 *     - [1]: sigma2s[num_competitors]: skill variance
 *   hyper_params: [beta, tau, epsilon]
 *   probs[num_matchups]: output probabilities for each match
 */
{
    // Dataset fields
    const int (*matchups)[2] = dataset.matchups;
    const double* outcomes = dataset.outcomes;
    const int num_matchups = dataset.num_matchups;
     
    // Model parameters
    double* mus = model_inputs.model_params[0];
    double* sigma2s = model_inputs.model_params[1];
    
    // Output array
    double* probs = model_inputs.probs;
    
    // Hyperparameters
    const double* h = model_inputs.hyper_params;
    const double beta = h[0], tau = h[1], epsilon = h[2];

    double c2, c, z, eps_over_c, w, v, step_a, step_b, sign;
    int idx_a, idx_b;
    const double tau2 = tau * tau;
    const double two_beta2 =  2.0 * beta * beta;

    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        sigma2s[idx_a] += tau2;
        sigma2s[idx_b] += tau2;
        c2 = two_beta2 + sigma2s[idx_a] + sigma2s[idx_b];
        c = sqrt(c2);
        eps_over_c = epsilon / c;
        z = (mus[idx_a] - mus[idx_b]) / c;
        probs[i] = norm_cdf(z);

        step_a = sigma2s[idx_a] / c;
        step_b = sigma2s[idx_b] / c;

        if (outcomes[i] != 0.5){
            sign = (2.0 * outcomes[i]) - 1.0; // map 1 -> 1, and 0 -> -1
            v_w_win(sign * z, eps_over_c, &v, &w);
        }
        else {
            sign = 1.0;
            v_w_draw(z, eps_over_c, &v, &w);
        }
        mus[idx_a] += sign * step_a * v;
        mus[idx_b] -= sign * step_b * v;

        // sigma2s[idx_a] -= step_a * step_a * w;
        // sigma2s[idx_b] -= step_b * step_b * w;
        sigma2s[idx_a] = fmax(1e-6, sigma2s[idx_a] - step_a * step_a * w);
        sigma2s[idx_b] = fmax(1e-6, sigma2s[idx_b] - step_b * step_b * w);
    }
}