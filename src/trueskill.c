#include <stdlib.h>
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


double* construct_trueskill_ratings(ModelInputs model_inputs)
{
    // Allocate memory for both mus and sigma2s in contiguous block
    double* memory = malloc(2 * model_inputs.num_competitors * sizeof(double));
    double* mus = memory;
    double* sigma2s = memory + model_inputs.num_competitors;
    
    double initial_mu = model_inputs.hyper_params[0];
    double initial_sigma = model_inputs.hyper_params[1];
    double initial_sigma2 = initial_sigma * initial_sigma;
    
    for (int i = 0; i < model_inputs.num_competitors; i++) {
        mus[i] = initial_mu;
        sigma2s[i] = initial_sigma2;
    }
    
    return memory;
}

ModelOutputs online_trueskill(Dataset dataset, ModelInputs model_inputs)
{
    const int (*matchups)[2] = dataset.matchups;
    const double* outcomes = dataset.outcomes;
    const int num_matchups = dataset.num_matchups;
    
    // Call initializer to get mus and sigma2s
    double* ratings_memory = construct_trueskill_ratings(model_inputs);
    double* mus = ratings_memory;
    double* sigma2s = ratings_memory + model_inputs.num_competitors;
    double* probs = malloc(num_matchups * sizeof(double));
    
    const double* h = model_inputs.hyper_params;
    const double beta = h[2], tau = h[3], epsilon = h[4];
    
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
    ModelOutputs model_outputs = {probs, ratings_memory};
    return model_outputs;
}