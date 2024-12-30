
import time
import math
import numpy as np
import polars as pl
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from ric import online_elo, online_glicko, online_trueskill, compute_metrics, sweep

def main():
    game = 'smash_melee'
    # game = 'league_of_legends'
    # game = 'tetris'
    df = load_dataset('EsportsBench/EsportsBench', split=game).to_polars().filter(pl.col('outcome') != 0.5)
    competitor_cols = ['competitor_1', 'competitor_2']

    # df = pl.read_csv('~/Downloads/chartslp.csv')
    # competitor_cols=['player_1_code', 'player_2_code']

    dataset = MatchupDataset(
        df=df,
        competitor_cols=competitor_cols,
        outcome_col='outcome',
        datetime_col='date',
        rating_period='1D'
    )

    matchups = dataset.matchups
    outcomes = dataset.outcomes
    time_steps = dataset.time_steps
    num_matchups = matchups.shape[0]
    num_competitors = len(dataset.competitors)

    # example from http://www.glicko.net/glicko/glicko.pdf
    # matchups = np.array(
    #     [[0,1],
    #      [1,2],
    #      [2,3],
    #      [3,0]],
    #      dtype=np.int32
    # )
    # outcomes = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    # num_matchups = 4
    # num_competitors = 4

    k = 32.0
    initial_rating = 1500.0
    scale = 400.0
    base = 10.0

    print('running Elo')
    start_time = time.time()
    ratings, probs = online_elo(
        matchups,
        outcomes,
        num_competitors,
        initial_rating,
        k,
        scale,
        base,
    )
    end_time = time.time()
    print(f'online Elo duration (s): {end_time-start_time:.4f}')
    print(f'ratings (min, max, mean): ({ratings.min():.4f}, {ratings.max():.4f}, {ratings.mean():.4f})')
    print(f'probs   (min, max, mean): ({probs.min():.4f}, {probs.max():.4f}, {probs.mean():.4f})')
    acc, log_loss, brier_score = compute_metrics(probs, outcomes)
    print(f'Elo (acc, log_loss, brier_score): {acc:.4f}, {log_loss:.4f}, {brier_score:.4f}')


    print('\nrunning Glicko')
    initial_r = 1500.0
    initial_rd = 350.0
    c = 63.2
    start_time = time.time()
    rs, rds, probs = online_glicko(
        matchups,
        time_steps,
        outcomes,
        num_competitors,
        initial_r,
        initial_rd,
        c,
        scale,
        base,
    )
    end_time = time.time()
    print(f'online Glicko duration (s): {end_time-start_time:.4f}')
    print(f'rs      (min, max, mean): ({rs.min():.4f}, {rs.max():.4f}, {rs.mean():.4f})')
    print(f'rds     (min, max, mean): ({rds.min():.4f}, {rds.max():.4f}, {rds.mean():.4f})')
    print(f'probs   (min, max, mean): ({probs.min():.4f}, {probs.max():.4f}, {probs.mean():.4f})')
    acc, log_loss, brier_score = compute_metrics(probs, outcomes)
    print(f'Glicko (acc, log_loss, brier_score): {acc:.4f}, {log_loss:.4f}, {brier_score:.4f}')


    print('\nrunning TrueSkill')
    # I tuned these a little bit
    initial_mu = 25.0
    initial_sigma = 2.0
    beta = 1.0
    tau = 0.25
    epsilon = 0.0
    start_time = time.time()
    mus, sigmas, probs = online_trueskill(
        matchups,
        outcomes,
        num_competitors,
        initial_mu,
        initial_sigma,
        beta,
        tau,
        epsilon
    )
    end_time = time.time()
    print(f'online TrueSkill duration (s): {end_time-start_time:.4f}')
    print(f'mus     (min, max, mean): ({mus.min():.4f}, {mus.max():.4f}, {mus.mean():.4f})')
    print(f'sigmas  (min, max, mean): ({sigmas.min():.4f}, {sigmas.max():.4f}, {sigmas.mean():.4f})')
    print(f'probs   (min, max, mean): ({probs.min():.4f}, {probs.max():.4f}, {probs.mean():.4f})')
    acc, log_loss, brier_score = compute_metrics(probs, outcomes)
    print(f'TrueSkill (acc, log_loss, brier_score): {acc:.4f}, {log_loss:.4f}, {brier_score:.4f}')

if __name__ == '__main__':
    main()