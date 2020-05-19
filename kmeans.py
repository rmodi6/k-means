import argparse

import numpy as np
import pandas as pd


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2), axis=1))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)


def cluster(X, y, k, distance):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    iteration = 0
    while True:
        iteration += 1
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}')
        clusters = {i: {'data': [], 'label': []} for i in range(k)}
        for i, Xi in enumerate(X):
            distances = distance(Xi, centroids)
            closest_cluster_id = np.argmin(distances)
            clusters[closest_cluster_id]['data'].append(Xi)
            clusters[closest_cluster_id]['label'].append(y[i])
        new_centroids = []
        for cluster_id in clusters:
            clusters[cluster_id]['data'] = np.array(clusters[cluster_id]['data'])
            new_centroids.append(np.mean(clusters[cluster_id]['data'], axis=0))
        new_centroids = np.array(new_centroids)
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return clusters


if __name__ == '__main__':
    distance_functions = {'euclidean': euclidean_distance, 'manhattan': manhattan_distance}

    parser = argparse.ArgumentParser(description="k-means clustering algorithm")

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset',
                        required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, help='number of clusters (default: 2)', default=2)
    parser.add_argument('--distance', dest='distance', choices=distance_functions.keys(), type=str.lower,
                        help='distance function (default: Euclidean)', default='Euclidean')

    args = parser.parse_args()
    distance_fn = distance_functions[args.distance]

    df = pd.read_csv(args.dataset_path)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    for run in range(3):
        print(f'Run: {run+1}')
        clusters = cluster(X, y, args.k, distance_fn)
        for cluster_id in clusters:
            print(f'Cluster id: {cluster_id}')
            pos_fraction = np.mean(clusters[cluster_id]['label'])
            print(f'Fraction of +ve labels: {pos_fraction:.2f}\t Fraction of -ve labels: {1 - pos_fraction:.2f}')
