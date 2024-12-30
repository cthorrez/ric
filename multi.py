
import time
import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from ric import sweep, sample_fit

def main():
    # game = 'smash_melee'
    game = 'league_of_legends'
    # game = 'tetris'
    df = load_dataset('EsportsBench/EsportsBench', split=game).to_polars()
    # df = df.filter(pl.col('outcome') != 0.5)
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
    num_samples = 1000
    replace = True
    num_threads = 16
    batch_size = 96
    plot = False

    print('running Bootstrap Elo')
    k = 32.0
    initial_rating = 1500.0
    scale = 400.0
    base = 10.0
    start_time = time.time()
    boot_ratings = sample_fit(
        'elo',
        matchups,
        None,
        outcomes,
        num_competitors,
        np.array([initial_rating, k, scale, base]),
        num_samples,
        replace=replace,
        num_threads=num_threads,
        batch_size=batch_size
    )
    end_time = time.time()
    print(f'bootstrap Elo duration (s): {end_time-start_time:.4f}')
    print(boot_ratings.shape)
    boot_ratings = boot_ratings.mean(axis=0)
    print(f'ratings (min, max, mean): ({boot_ratings.min():.4f}, {boot_ratings.max():.4f}, {boot_ratings.mean():.4f})')
    plt.hist(boot_ratings)
    plt.title('Elo Rating Distribution')
    if plot: plt.show()

    print('\nrunning Bootstrap Glicko')
    initial_r = 1500.0
    initial_rd = 350.0
    c = 0.06
    scale = 200.0
    base = 10.0
    start_time = time.time()
    boot_ratings = sample_fit(
        'glicko',
        matchups,
        time_steps,
        outcomes,
        num_competitors,
        np.array([initial_r, initial_rd, c, scale, base]),
        num_samples,
        replace=replace,
        num_threads=num_threads,
        batch_size=batch_size
    )
    end_time = time.time()
    print(f'bootstrap Glicko duration (s): {end_time-start_time:.4f}')
    print(boot_ratings.shape)
    boot_ratings = boot_ratings.mean(axis=0)
    print(f'ratings (min, max, mean): ({boot_ratings.min():.4f}, {boot_ratings.max():.4f}, {boot_ratings.mean():.4f})')
    plt.hist(boot_ratings)
    plt.title('Glicko Rating Distribution')
    if plot: plt.show()



    print('\nrunning Bootstrap TrueSkill')
    mu = 25.0
    sigma = mu / 3
    beta = sigma / 2
    tau = sigma / 100
    epsilon = 1e-4
    start_time = time.time()
    boot_ratings = sample_fit(
        'trueskill',
        matchups,
        None,
        outcomes,
        num_competitors,
        np.array([mu, sigma, beta, tau, epsilon]),
        num_samples,
        replace=replace,
        num_threads=num_threads,
        batch_size=batch_size
    )
    end_time = time.time()
    print(f'bootstrap TrueSkill duration (s): {end_time-start_time:.4f}')
    print(boot_ratings.shape)
    boot_ratings = boot_ratings.mean(axis=0)
    print(f'ratings (min, max, mean): ({boot_ratings.min():.4f}, {boot_ratings.max():.4f}, {boot_ratings.mean():.4f})')
    plt.hist(boot_ratings)
    plt.title('TrueSkill Rating Distribution')
    if plot: plt.show()

if __name__ == '__main__':
    main()