import argparse

import numpy as np
import pandas as pd


def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1 - X2), axis=1))


def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1 - X2), axis=1)


if __name__ == '__main__':
    distance_functions = {'euclidean': euclidean_distance, 'manhattan': manhattan_distance}

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset',
                        required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, help='number of clusters', default=2)
    parser.add_argument('--distance', dest='distance', choices=distance_functions.keys(),
                        type=str.lower, help='', default='Euclidean')

    args = parser.parse_args()
    distance_fn = distance_functions[args.distance]

    df = pd.read_csv(args.dataset_path)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values