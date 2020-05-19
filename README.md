# k-means clustering
k-means clustering algorithm implemented in Python.

### Usage
```bash
usage: kmeans.py [-h] --dataset DATASET_PATH [--k K]
                 [--distance {euclidean,manhattan}]

k-means clustering algorithm

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_PATH
                        path to dataset
  --k K                 number of clusters (default: 2)
  --distance {euclidean,manhattan}
                        distance function (default: Euclidean)

```

### Example
```bash
python kmeans.py --dataset dataset/Breast_cancer_data.csv --k 2 --distance manhattan
```
