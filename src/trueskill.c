#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdbool.h>  // Add this with your other includes
#include "ric.h"


#define M_1_SQRT_2PI 0.39894228040143267793994605993438
double norm_pdf(double z){
    return M_1_SQRT_2PI * exp(-0.5 * z * z);
}

double norm_cdf(double z){
    return 0.5 * (1.0 + erf(z * M_SQRT1_2));
}

void v_w_win(double t, double epsilon, double* v, double* w) {
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

void v_w_draw(double t, double epsilon, double* v, double* w){
    double e_p_t = epsilon + t;
    double e_m_t = epsilon - t;
    double denom = norm_cdf(e_m_t) + norm_cdf(e_p_t) - 1.0; // this relies on cdf(-epsilon - t) = 1 - cdf(epsilon + t)
    double pdf_m = norm_pdf(e_m_t);
    double pdf_p = norm_pdf(e_p_t); // this relies on pdf(epsilon + t) = pdf(-epsilon - t)
    *v = (pdf_p - pdf_m) / denom;
    *w = ((e_m_t * pdf_m) + (e_p_t * pdf_p)) / denom;
}


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
    double c2, c, z, eps_over_c, w, v, step_a, step_b, sign;
    int idx_a, idx_b;
    double tau2 = tau * tau;
    double two_beta2 =  2.0 * beta * beta;

    for (int i = 0; i < num_matchups; i++) {
        idx_a = matchups[i][0];
        idx_b = matchups[i][1];
        var[idx_a] += tau2;
        var[idx_b] += tau2;
        c2 = two_beta2 + var[idx_a] + var[idx_b];
        c = sqrt(c2);
        eps_over_c = epsilon / c;
        z = (mean[idx_a] - mean[idx_b]) / c;
        probs[i] = norm_cdf(z);

        step_a = var[idx_a] / c;
        step_b = var[idx_b] / c;

        if (outcomes[i] != 0.5){
            sign = (2.0 * outcomes[i]) - 1.0; // map 1 -> 1, and 0 -> -1
            v_w_win(sign * z, eps_over_c, &v, &w);
        }
        else {
            sign = 1.0;
            v_w_draw(z, eps_over_c, &v, &w);
        }
        mean[idx_a] += sign * step_a * v;
        mean[idx_b] -= sign * step_b * v;

        // var[idx_a] -= step_a * step_a * w;
        // var[idx_b] -= step_b * step_b * w;
        var[idx_a] = fmax(1e-6, var[idx_a] - step_a * step_a * w);
        var[idx_b] = fmax(1e-6, var[idx_b] - step_b * step_b * w);
    }
}