#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <stdio.h>
#include "ric.h"

ModelOutputs* multi_dataset_fit(
    RatingSystem model,
    Dataset* datasets,
    ModelInputs model_inputs,
    int num_runs,
    int num_threads
) {
    ModelOutputs* outputs = (ModelOutputs*) malloc(sizeof(ModelOutputs) * num_runs);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_runs; i++) {
        outputs[i] = model(datasets[i], model_inputs);
    }
    return outputs;
}

ModelOutputs* multi_params_fit(
    RatingSystem model,
    Dataset dataset,
    ModelInputs* model_inputs,
    int num_runs,
    int num_threads
) {
    ModelOutputs* outputs = (ModelOutputs*) malloc(sizeof(ModelOutputs) * num_runs);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_runs; i++) {
        outputs[i] = model(dataset, model_inputs[i]);
    }
    return outputs;
}