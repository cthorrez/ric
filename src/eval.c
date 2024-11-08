#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
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

double* evaluate(
    RatingSystem model,
    Dataset dataset,
    ModelInputs model_inputs,
    double metrics[3]
)
{
    ModelOutputs model_outputs = model(dataset, model_inputs);
    compute_metrics(model_outputs.probs, dataset.outcomes, metrics, dataset.num_matchups);
    free(model_outputs.probs);
    free(model_outputs.ratings);
    return metrics;
}

SweepOutputs sweep(
    RatingSystem model,
    Dataset dataset,
    ModelInputs* sweep_inputs,
    int num_sweep_inputs
) {
    omp_set_num_threads(24);
    double best_metric = INFINITY;  // Start with worst possible for minimization
    double* best_metrics = malloc(3 * sizeof(double));
    ModelInputs best_inputs = sweep_inputs[0];
    
    #pragma omp parallel
    {
        double local_best_metric = INFINITY;  // Start with worst possible for minimization
        double local_metrics[3];
        ModelInputs local_best_inputs;
        double local_best_metrics[] = {0.0, INFINITY, INFINITY};
        
        #pragma omp for
        for (int i = 0; i < num_sweep_inputs; i++) {
            evaluate(model, dataset, sweep_inputs[i], local_metrics);
            // Using log loss (metrics[1]) as the optimization target
            // Check for numerical stability and improvement
            if ((local_metrics[1] < local_best_metric) && 
                (local_metrics[1] > 0) && 
                (local_metrics[1] < 1000)) {
                local_best_metric = local_metrics[1];  // Using log loss as primary metric
                for(int j = 0; j < 3; j++) {
                    local_best_metrics[j] = local_metrics[j];
                }
                local_best_inputs = sweep_inputs[i];
            }
        }
        
        #pragma omp critical
        {
            if (local_best_metric < best_metric) {  // Consistent minimization
                best_metric = local_best_metric;
                for(int j = 0; j < 3; j++) {
                    best_metrics[j] = local_best_metrics[j];
                }
                best_inputs = local_best_inputs;
            }
        }
    }
    
    SweepOutputs outputs = {best_metrics, best_inputs};
    return outputs;
}