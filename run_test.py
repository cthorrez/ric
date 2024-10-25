
import time
import numpy as np
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from ric import run_elo

def main():

    game = 'league_of_legends'
    game = 'tetris'
    df = load_dataset('EsportsBench/EsportsBench', split=game).to_pandas()
    dataset = MatchupDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period='7D'
    )
    matchups = dataset.matchups.astype(np.int32)
    outcomes = dataset.outcomes
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


    initial_rating = 1500.0
    scale = 400.0
    base = 10.0

    start_time = time.time()
    ratings = run_elo(
        matchups,
        outcomes,
        num_matchups,
        num_competitors,
        initial_rating,
        scale,
        base
    )
    end_time = time.time()
    print(f'duration (s): {end_time-start_time:.4f}')
    print(ratings.shape)



if __name__ == '__main__':
    main()