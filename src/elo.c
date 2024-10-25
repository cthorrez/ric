#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ric.h"

double* run_elo(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
    int num_matchups,         // Number of matchups
    int num_competitors,      // Number of competitors
    double initial_rating,    // Initial Elo rating for all competitors
    double scale,             // Elo scale factor
    double base               // Base rating change factor
)
{
    // initialize ratings
    double *ratings = (double *)malloc(num_competitors * sizeof(double));
    for (int i = 0; i < num_competitors; i++) {
        ratings[i] = initial_rating;
    }

    double alpha = log(base) / scale;
    for (int i = 0; i < num_matchups; i++) {
        int idx_a = matchups[i][0];
        int idx_b = matchups[i][1];
        double delta_r = ratings[idx_b] - ratings[idx_a];
        double prob = 1.0 / (1.0 + exp(alpha * delta_r));
        double update = outcomes[i] - prob;
        ratings[idx_a] += 32.0 * update;
        ratings[idx_b] -= 32.0 * update;
    }
    printf("done!\n");
    return ratings;
}