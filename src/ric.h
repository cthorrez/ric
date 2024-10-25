#ifndef RIC_H
#define RIC_H

void run_elo(
    int matchups[][2],        // 2D array for matchups
    double outcomes[],        // Array of outcomes (e.g., win/loss/draw)
    int num_matchups,         // Number of matches
    int num_competitors,      // Number of competitors
    double initial_rating,    // Initial Elo rating for all competitors
    double k,                 // K controls the size of the updates
    double scale,             // Elo scale factor
    double base,              // Log base
    double ratings[],
    double probs[]
);

#endif