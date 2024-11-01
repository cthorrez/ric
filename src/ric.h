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
    Dataset* dataset;
    double** model_params;
    double* hyper_params;
    double* probs;
} ModelInputs;

void online_elo(ModelInputs model_inputs);
void online_glicko(ModelInputs model_inputs);
void online_trueskill(ModelInputs model_inputs);

double* compute_metrics(
    double probs[],
    double outcomes[],
    double metrics[3],
    int n
);

typedef double (*RatingSystem)(double* params);
void param_sweep(
    RatingSystem model,      // function pointer to model
    double** param_sets,     // array of parameter sets [n_trials][n_params]
    int n_trials,           // number of trials (100 in your case)
    int n_params,           // number of params this model uses
    double* best_params,    // output: best params found
    double* best_metric     // output: best metric found
);

#endif