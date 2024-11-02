#include <math.h>
#include <stdbool.h>
#include "ric.h"

void compute_metrics(
    double probs[],
    double outcomes[],
    double metrics[3],
    int n
) 
{
    for (int i=0; i<n; i++) {
        metrics[0] += (double)((probs[i] >= 0.5) == (bool)outcomes[i]);
        metrics[1] += -(outcomes[i] * log(probs[i])) - ((1.0 - outcomes[i]) * log(1.0 - probs[i]));
        metrics[2] += (probs[i] - outcomes[i]) * (probs[i] - outcomes[i]);

    }
    metrics[0] = metrics[0] / n;
    metrics[1] = metrics[1] / n;
    metrics[2] = metrics[2] / n;
}

double evaluate(
    RatingSystem model,
    ModelInputs model_inputs,
    double metrics[3]
)
{
    model(model_inputs);
    compute_metrics(model_inputs.probs, model_inputs.dataset->outcomes, metrics, model_inputs.dataset->num_matchups);
    return metrics[0];
}


// void param_sweep(
//     RatingSystem model,      // function pointer to model
//     double** param_sets,     // array of parameter sets [n_trials][n_params]
//     int n_trials,           // number of trials (100 in your case)
//     int n_params,           // number of params this model uses
//     double* best_params,    // output: best params found
//     double* best_metric     // output: best metric found
// ) {
//     *best_metric = INFINITY;  // assuming higher is better
    
//     for (int trial = 0; trial < n_trials; trial++) {
//         double metric = model(param_sets[trial]);
//         if (metric < *best_metric) {
//             *best_metric = metric;
//             memcpy(best_params, param_sets[trial], n_params * sizeof(double));
//         }
//     }
// }