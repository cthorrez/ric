
import time
import numpy as np
import polars as pl
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from ric import online_elo, online_glicko, online_trueskill

def main():
    # game = 'smash_melee'
    # game = 'league_of_legends'
    # game = 'tetris'
    # df = load_dataset('EsportsBench/EsportsBench', split=game).to_polars()
    # competitor_cols = ['competitor_1', 'competitor_2']

    df = pl.read_csv('~/Downloads/chartslp.csv')
    competitor_cols=['player_1_code', 'player_2_code']

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


    start_time = time.time()
    ratings, probs = online_elo(
        matchups,
        outcomes,
        num_matchups,
        num_competitors,
        # initial_rating,
        # k,
        # scale,
        # base,
    )
    end_time = time.time()
    print(f'online elo duration (s): {end_time-start_time:.4f}')
    print(ratings.shape)
    sort_idxs = np.argsort(-ratings)
    for idx in sort_idxs[:10]:
        c = dataset.competitors[idx]
        print(f'{c[:20]:20}:{ratings[idx]:.4f}')

    acc = ((probs >= 0.5) == outcomes).astype(np.float64).mean()
    print(acc)

    initial_r, initial_rd = 1500.0, 350
    c = 63.2
    rs, rds, probs = online_glicko(
        matchups,
        time_steps,
        outcomes,
        num_matchups,
        num_competitors,
        # initial_r,
        # initial_rd,
        # c,
        # scale,
        # base,
    )
    end_time = time.time()
    print(f'online glicko duration (s): {end_time-start_time:.4f}')

    print(f'{rs.min()=}, {rs.max()=}, {rs.mean()=}')
    print(f'{rds.min()=}, {rds.max()=}, {rds.mean()=}')
    print(f'{probs.min()=}, {probs.max()=}, {probs.mean()=}')
    print(probs)
    acc = ((probs >= 0.5) == outcomes).astype(np.float64).mean()
    print(acc)

    mus, sigmas, probs = online_trueskill(
        matchups,
        outcomes,
        num_matchups,
        num_competitors,
    )
    # print(mus)
    # print(sigmas)
    



if __name__ == '__main__':
    main()