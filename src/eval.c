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
    int num_sweep_inputs,
    int num_threads
) {
    omp_set_num_threads(num_threads);
    
    // Allocate arrays to store results for each input
    double* all_metrics = malloc(num_sweep_inputs * 3 * sizeof(double));
    
    #pragma omp parallel for
    for (int i = 0; i < num_sweep_inputs; i++) {
        // Each thread writes to its own location in the array
        evaluate(model, dataset, sweep_inputs[i], &all_metrics[i * 3]);
    }
    
    // Find best result in a single pass after parallel section
    int best_idx = 0;
    double best_log_loss = INFINITY;
    
    for (int i = 0; i < num_sweep_inputs; i++) {
        double log_loss = all_metrics[i * 3 + 1];  // metrics[1] is log loss
        if ((log_loss < best_log_loss) && (log_loss > 0) && (log_loss < 1000)) {
            best_log_loss = log_loss;
            best_idx = i;
        }
    }
    
    // Allocate and copy best metrics
    double* best_metrics = malloc(3 * sizeof(double));
    for (int j = 0; j < 3; j++) {
        best_metrics[j] = all_metrics[best_idx * 3 + j];
    }
    
    ModelInputs best_inputs = sweep_inputs[best_idx];
    
    // Clean up
    free(all_metrics);
    
    SweepOutputs outputs = {best_metrics, best_inputs};
    return outputs;
}