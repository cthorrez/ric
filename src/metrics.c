#include <math.h>
#include <stdbool.h>
#include "ric.h"

double* compute_metrics(
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
    return metrics;
}