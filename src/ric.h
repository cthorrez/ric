#ifndef RIC_H
#define RIC_H

void online_elo(
    const int matchups[][2],
    const double outcomes[],
    double mean[],
    double probs[],
    const int num_matchups,
    const int num_competitors,
    const double k,
    const double scale,
    const double base
);

void online_glicko(
    const int matchups[][2],
    const int time_steps[],
    const double outcomes[],
    double mean[],
    double var[],
    double probs[],
    const int num_matchups,
    const int num_competitors,
    const double max_rd,
    const double c,
    const double scale,
    const double base
);

void online_trueskill(
    const int matchups[][2],
    const double outcomes[],
    double mean[],
    double var[],
    double probs[],
    const int num_matchups,
    const int num_competitors,
    const double beta,
    const double tau,
    const double epsilon
);

double* compute_metrics(
    double probs[],
    double outcomes[],
    double metrics[3],
    int n
);

#endif