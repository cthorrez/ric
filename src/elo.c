#include <stdlib.h>
#include <math.h>
#include "ric.h"

double* run_elo(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
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

    return ratings;
}