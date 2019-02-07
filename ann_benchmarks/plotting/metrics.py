from __future__ import absolute_import
import numpy as np

def knn(dataset_distances, run_distances, count, attrs, epsilon=1e-10):
    if 'knn' not in attrs:
        total = len(run_distances) * count
        recalls = np.zeros(len(run_distances))
        for i in range(len(run_distances)):
            t = dataset_distances[i][count - 1] + epsilon
            actual = 0
            for j in range(count):
                if run_distances[i][j] <= t:
                    actual += 1
                else:
                    break
            recalls[i] = actual
        attrs['knn'] = np.mean(recalls) / float(count), np.std(recalls) / float(count)
    else:
        print("Found result")
    return attrs['knn']


def epsilon(dataset_distances, run_distances, count, attrs, epsilon=0.01):
    s = 'eps' + str(epsilon)
    if s not in attrs:
        total = len(run_distances) * count
        recalls = np.zeros(len(run_distances))
        for i in range(len(run_distances)):
            t = dataset_distances[i][count - 1] * (1 + epsilon)
            actual = 0
            for j in range(count):
                if run_distances[i][j] <= t:
                    actual += 1
                else:
                    break
            recalls[i] = actual
        attrs[s] = np.mean(recalls) / float(count), np.std(recalls) / float(count)
    return attrs[s]

def rel(dataset_distances, run_distances, attrs):
    if 'rel' not in attrs:
        total_closest_distance = 0.0
        total_candidate_distance = 0.0
        for true_distances, found_distances in zip(dataset_distances, run_distances):
            for rdist, cdist in zip(true_distances, found_distances):
                total_closest_distance += rdist
                total_candidate_distance += cdist
        if total_closest_distance < 0.01:
            return float("inf")
        attrs['rel'] = total_candidate_distance / total_closest_distance
    return attrs['rel']

def queries_per_second(query_times, attrs):
    return 1.0 / np.mean(query_times), 1.0 / np.std(query_times)

def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)

def build_time(queries, attrs):
    return attrs["build_time"]

def candidates(queries, attrs):
    return attrs["candidates"]

def dist_comps(queries, attrs):
    return attrs.get("dist_comps", 0) / len(queries)

all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, query_times, run_attrs: knn(true_distances, run_distances, run_attrs["count"], run_attrs)[0],
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "k-nn-std": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, query_times, run_attrs: knn(true_distances, run_distances, run_attrs["count"], run_attrs)[1],
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, query_times, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], run_attrs)[0],
        "worst": float("-inf")
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, query_times, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], run_attrs, 0.1)[0],
        "worst": float("-inf")
    },
    "rel": {
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, query_times, run_attrs: rel(true_distances, run_distances, run_attrs),
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, query_times, run_attrs: queries_per_second(query_times, run_attrs)[0],
        "worst": float("-inf")
    },
    "qps-dev": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, query_times, run_attrs: queries_per_second(query_times, run_attrs)[1],
        "worst": float("-inf")
    },
    "distcomps" : {
        "description": "Distance computations",
        "function": lambda true_distances, run_distances, query_times, run_attrs: dist_comps(true_distances, run_attrs),
        "worst": float("inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_distances, run_distances, query_times, run_attrs: build_time(true_distances, run_attrs),
        "worst": float("inf")
    },
    "candidates" : {
        "description": "Candidates generated",
        "function": lambda true_distances, run_distances, query_times, run_attrs: candidates(true_distances, run_attrs),
        "worst": float("inf")
    },
    "indexsize" : {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, query_times, run_attrs: index_size(true_distances, run_attrs),
        "worst": float("inf")
    },
    "queriessize" : {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_distances, run_distances, query_times, run_attrs: index_size(true_distances, run_attrs) / queries_per_second(true_distances, run_attrs)[0],
        "worst": float("inf")
    }
}
