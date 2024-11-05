#ifndef RIC_H
#define RIC_H

typedef struct {
    int (*matchups)[2];
    int* time_steps;
    double* outcomes;
    int num_matchups;
    int num_competitors;
} Dataset;

typedef struct{
    Dataset dataset;
    double** model_params;
    double* hyper_params;
    double* probs;
} ModelInputs;

void online_elo(ModelInputs model_inputs);
void online_glicko(ModelInputs model_inputs);
void online_trueskill(ModelInputs model_inputs);

typedef void (*RatingSystem)(ModelInputs model_inputs);

void compute_metrics(
    double probs[],
    double outcomes[],
    double metrics[3],
    int n
);

double evaluate(
    RatingSystem model,
    ModelInputs model_inputs,
    double metrics[3]
);


void param_sweep(
    RatingSystem model,
    Dataset dataset,
    double** param_sets,
    int n_trials,
    int n_params,
    double* best_params,
    double* best_metric
);

#endif