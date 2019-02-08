import pandas as pd
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms.definitions import get_definitions
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils  import get_plot_label, compute_metrics_all_runs, compute_metrics, create_linestyles, create_pointset
from ann_benchmarks.results import store_results, load_all_results, get_unique_algorithms, get_algorithm_name

datasets = [
    'fashion-mnist-784-euclidean-easy',
    'fashion-mnist-784-euclidean-hard',
    'fashion-mnist-784-euclidean-middle',
    'glove-100-angular-easy',
    'glove-100-angular-hard',
    'glove-100-angular-middle',
    'mnist-784-euclidean-easy',
    'mnist-784-euclidean-hard',
    'mnist-784-euclidean-middle',
    'random-xs-20-euclidean',
    'sift-128-euclidean-easy',
    'sift-128-euclidean-hard',
    'sift-128-euclidean-middle',
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--dataset',
    #     metavar="DATASET",
    #     default='glove-100-angular')
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '--batch',
        help='Plot runs in batch mode',
        action='store_true')
    parser.add_argument(
        '--output',
        help='Path to the output csv file')
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Path to the output csv file')
    args = parser.parse_args()

    count = int(args.count)
    dataframes = []
    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        dataset = get_dataset(dataset_name)
        unique_algorithms = get_unique_algorithms()
        results = load_all_results(dataset_name, count, True, args.batch)
        results = compute_metrics_all_runs(list(dataset["distances"]), results, args.recompute)
        data = pd.DataFrame(results)
        data['dataset'] = dataset_name
        data['count'] = count
        dataframes.append(data)
    data = pd.concat(dataframes)
    print(data.groupby(['dataset', 'count', 'algorithm', 'parameters']).count())
    with open(args.output, 'w') as fp:
        data.to_csv(fp)



