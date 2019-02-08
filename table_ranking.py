import os
import sys
import matplotlib
# I get a crash without this line
matplotlib.use('Agg')
import numpy as np
from plotnine import *
import pandas as pd
import re

def select_best(group):
    idx = group['score'].idxmax()
    best = group.loc[idx]
    del best['algorithm']
    del best['dataset']
    return best

def sort_dataset(group):
    return group.sort_values(by='score', ascending=False)

def build_table(recall, path):
    data = pd.read_csv(sys.argv[1])
    data['score'] = np.cbrt(data['qps'] * (1.0/data['indexsize']) * (1.0/data['distcomps']))
    data = data[data['k-nn'] >= recall]
    best = data.groupby(['dataset', 'algorithm']).apply(select_best)
    best = best.groupby(['dataset']).apply(sort_dataset).reset_index(0, drop=True)
    return best

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python3 table_ranking.py RESULTS_FILE knn")
        sys.exit()
    recall = float(sys.argv[2])
    best = build_table(recall, sys.argv[1])
    print(best[['k-nn', 'score', 'qps', 'indexsize', 'distcomps']])


