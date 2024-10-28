#ifndef RIC_H
#define RIC_H

void online_elo(
    int matchups[][2],
    double outcomes[],
    double mean[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double k,
    double scale,
    double base
);

void online_glicko(
    int matchups[][2],
    int time_steps[],
    double outcomes[],
    double mean[],
    double var[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double max_rd,
    double c,
    double scale,
    double base
);


void online_trueskill(
    int matchups[][2],
    double outcomes[],
    double mean[],
    double var[],
    double probs[],
    int num_matchups,
    int num_competitors,
    double beta,
    double tau,
    double epsilon
);

#endif