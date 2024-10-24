#ifndef RIC_H
#define RIC_H

double* run_elo(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
    int num_competitors,      // Number of competitors
    double initial_rating,    // Initial Elo rating for all competitors
    double scale,             // Elo scale factor
    double base               // Base rating change factor
);

#endif