#ifndef RIC_H
#define RIC_H

void online_elo(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
    int num_matchups,         // Number of matches
    int num_competitors,      // Number of competitors
    double k,                 // K controls the size of the updates
    double scale,             // Elo scale factor
    double base,              // Log base
    double ratings[],
    double probs[]
);

void online_glicko(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
    int time_steps[],         // Array of time_steps
    int num_matchups,         // Number of matches
    int num_competitors,      // Number of competitors
    double initial_rd,
    double c,                 // increase in rd per unit time
    double scale,             // Elo scale factor
    double base,              // Log base
    double rs[],
    double rds[],
    double probs[]
);

#endif