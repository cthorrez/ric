
import time
import math
import numpy as np
import polars as pl
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from ric import online_elo, online_glicko, online_trueskill, compute_metrics, sweep, sweep_batch_eval

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

    rng = np.random.default_rng(seed=0)
    num_sweep_inputs = 1000
    num_threads = 24
    elo_sweep_inputs = np.empty((num_sweep_inputs,4))
    elo_sweep_inputs[:,0] = np.full(shape=(num_sweep_inputs), fill_value=1500) # in elo the initial rating does not matter
    elo_sweep_inputs[:,1] = rng.uniform(16.0, 64.0, size=(num_sweep_inputs,)) # ks
    elo_sweep_inputs[:,2] = rng.uniform(100.0, 400.0, size=(num_sweep_inputs,)) # scales
    elo_sweep_inputs[:,3] = rng.uniform(math.e / 2, math.e *2, size=(num_sweep_inputs,)) # bases

    print(f'Sweeping Elo over {num_sweep_inputs} hyperparameter combinations')
    start_time = time.time()
    best_metrics, best_params = sweep(
        system_name="elo",
        matchups=matchups,
        time_steps=None, # Elo does not use time info
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=elo_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')

    print(f'Best metrics (acc, log_loss, brier_score): ({best_metrics[0]:.4f}, {best_metrics[1]:.4f}, {best_metrics[2]:.4f})')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  k: {best_params[1]}")
    print(f"  scale: {best_params[2]}")
    print(f"  base: {best_params[3]}")


    glicko_sweep_inputs = np.empty((num_sweep_inputs, 5))
    glicko_sweep_inputs[:,0] = np.full(shape=(num_sweep_inputs), fill_value=1500) # in elo the initial rating does not matter
    glicko_sweep_inputs[:,1] = rng.uniform(10.0, 500.0, size=(num_sweep_inputs,)) # initial/max rd
    glicko_sweep_inputs[:,2] = rng.uniform(6.0, 100.0, size=(num_sweep_inputs,)) # c
    glicko_sweep_inputs[:,3] = rng.uniform(100.0, 400.0, size=(num_sweep_inputs,)) # scales
    glicko_sweep_inputs[:,4] = rng.uniform(math.e / 2, math.e *2, size=(num_sweep_inputs,)) # bases

    print(f'Sweeping Glicko over {num_sweep_inputs} hyperparameter combinations')
    start_time = time.time()
    best_metrics, best_params = sweep(
        system_name="glicko",
        matchups=matchups,
        time_steps=time_steps, # time is needed for Glicko
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=glicko_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')

    print(f'Best metrics (acc, log_loss, brier_score): ({best_metrics[0]:.4f}, {best_metrics[1]:.4f}, {best_metrics[2]:.4f})')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  Initial/max RD: {best_params[1]}")
    print(f"  c: {best_params[2]}")
    print(f"  scale: {best_params[3]}")
    print(f"  base: {best_params[4]}")


    trueskill_sweep_inputs = np.empty((num_sweep_inputs, 5))
    trueskill_sweep_inputs[:,0] = np.full(shape=(num_sweep_inputs), fill_value=25.0) # the initial rating does not matter
    trueskill_sweep_inputs[:,1] = rng.uniform(1.0, 16.0, size=(num_sweep_inputs,)) # initial sigma2
    trueskill_sweep_inputs[:,2] = rng.uniform(1.0, 16.0, size=(num_sweep_inputs,)) # beta
    trueskill_sweep_inputs[:,3] = rng.uniform(0.001, 0.5, size=(num_sweep_inputs,)) # tau
    trueskill_sweep_inputs[:,4] = rng.uniform(1e-6, 1e-3, size=(num_sweep_inputs,)) # epsilon

    print(f'Sweeping TrueSkill over {num_sweep_inputs} hyperparameter combinations')
    start_time = time.time()
    best_metrics, best_params = sweep(
        system_name="trueskill",
        matchups=matchups,
        time_steps=None,  # not needed for TrueSkill
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=trueskill_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')

    print(f'Best metrics (acc, log_loss, brier_score): ({best_metrics[0]:.4f}, {best_metrics[1]:.4f}, {best_metrics[2]:.4f})')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  Initial sigma²: {best_params[1]}")
    print(f"  beta: {best_params[2]}")
    print(f"  tau: {best_params[3]}")
    print(f"  epsilon: {best_params[4]}")


    print('\nRunning Batch Eval Sweep for Elo')
    start_time = time.time()
    best_ratings, best_params = sweep_batch_eval(
        system_name="elo",
        matchups=matchups,
        time_steps=None, # Elo does not use time info
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=elo_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  k: {best_params[1]}")
    print(f"  scale: {best_params[2]}")
    print(f"  base: {best_params[3]}")

    print('\nRunning Batch Eval Sweep for Glicko')
    start_time = time.time()
    best_ratings, best_params = sweep_batch_eval(
        system_name="glicko",
        matchups=matchups,
        time_steps=time_steps, # time is needed for Glicko
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=glicko_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  Initial/max RD: {best_params[1]}")
    print(f"  c: {best_params[2]}")
    print(f"  scale: {best_params[3]}")
    print(f"  base: {best_params[4]}")


    print('\nRunning Batch Eval Sweep for TrueSkill')
    start_time = time.time()
    best_ratings, best_params = sweep_batch_eval(
        system_name="trueskill",
        matchups=matchups,
        time_steps=None,  # not needed for TrueSkill
        outcomes=outcomes,
        num_competitors=num_competitors,
        param_sets=trueskill_sweep_inputs,
        num_threads=num_threads,
    )
    duration = time.time() - start_time
    print(f'sweep duration (s): {duration:.4f}')
    print("Best parameters found:")
    print(f"  Initial rating: {best_params[0]}")
    print(f"  Initial sigma²: {best_params[1]}")
    print(f"  beta: {best_params[2]}")
    print(f"  tau: {best_params[3]}")
    print(f"  epsilon: {best_params[4]}")




if __name__ == '__main__':
    main()


