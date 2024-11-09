#ifndef RIC_H
#define RIC_H

typedef struct {
    int (*matchups)[2];
    int* time_steps;
    double* outcomes;
    int num_matchups;
} Dataset;

typedef struct{
    double* hyper_params;
    int num_competitors;
} ModelInputs;

typedef struct{
    double* probs;
    double* ratings;
} ModelOutputs;

typedef struct{
    double* best_metrics;
    ModelInputs best_inputs;
} SweepOutputs;

ModelOutputs online_elo(Dataset dataset, ModelInputs model_inputs);
ModelOutputs online_glicko(Dataset dataset, ModelInputs model_inputs);
ModelOutputs online_trueskill(Dataset dataset, ModelInputs model_inputs);

typedef ModelOutputs (*RatingSystem)(Dataset dataset, ModelInputs model_inputs);

void compute_metrics(
    double probs[],
    double outcomes[],
    double metrics[3],
    int n
);

double* evaluate(
    RatingSystem model,
    Dataset dataset,
    ModelInputs model_inputs,
    double metrics[3]
);

SweepOutputs sweep(
    RatingSystem model,
    Dataset dataset,
    ModelInputs* sweep_inputs,
    int num_sweep_inputs,
    int num_threads
);

#endif