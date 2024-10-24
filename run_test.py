import numpy as np
from ric import calculate_elo

def main():
    matchups = np.array(
        [[0,1],
         [1,2],
         [2,3],
         [3,0]],
         dtype=np.int32
    )
    outcomes = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    num_competitors = 4
    initial_rating = 1500.0
    scale = 400.0
    base = 10.0

    ratings = calculate_elo(
        matchups,
        outcomes,
        num_competitors,
        initial_rating,
        scale,
        base
    )
    print(ratings)



if __name__ == '__main__':
    main()