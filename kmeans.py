import argparse

import numpy as np
import pandas as pd


def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1 - X2), axis=1))


def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1 - X2), axis=1)


def cluster(X, k, distance):
    clusters = {i: [] for i in range(k)}
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    iteration = 0
    while True:
        iteration += 1
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}')
        for Xi in X:
            distances = distance(Xi, centroids)
            clusters[np.argmin(distances)].append(Xi)
        new_centroids = []
        for cluster_id in clusters:
            new_centroids.append(np.mean(np.array(clusters[cluster_id]), axis=0))
        new_centroids = np.array(new_centroids)
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return clusters


if __name__ == '__main__':
    distance_functions = {'euclidean': euclidean_distance, 'manhattan': manhattan_distance}

    parser = argparse.ArgumentParser(description="kmeans clustering algorithm")

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset',
                        required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, help='number of clusters (default: 2)', default=2)
    parser.add_argument('--distance', dest='distance', choices=distance_functions.keys(), type=str.lower,
                        help='distance function (default: Euclidean)', default='Euclidean')

    args = parser.parse_args()
    distance_fn = distance_functions[args.distance]

    df = pd.read_csv(args.dataset_path)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    clusters = cluster(X, args.k, distance_fn)
