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
    Dataset dataset,
    ModelInputs model_inputs,
    double metrics[3]
)
{
    ModelOutputs model_outputs = model(dataset, model_inputs);
    compute_metrics(model_outputs.probs, dataset.outcomes, metrics, dataset.num_matchups);
    return metrics[0];
}


// void param_sweep(
//     RatingSystem model,
//     Dataset dataset,
//     double** param_sets,
//     int n_trials,
//     int n_params,
//     double* best_params,
//     double* best_metric
// ) {
//     *best_metric = INFINITY;  // assuming higher is better
    
//     for (int trial = 0; trial < n_trials; trial++) {
//         // TODO: instantiate the proper ModelInputs
//     }
// }