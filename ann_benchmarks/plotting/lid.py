import sys
import matplotlib
# I get a crash without this line
matplotlib.use('Agg')
from plotnine import *
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

def lid(path):
    """Plot the distribution of the LID score from the given dataset"""
    data = pd.read_csv(path, sep=' ', names=['query', 'lid'])

    percentiles = np.percentile(data.lid, (5,50,95))
    # Use the default mapping, unless we sample things. In that case we will
    # add the weight to the mapping
    geom_density_mapping = aes()

    # if it's too big, sample it
    threshold = 200000
    num_to_keep = 1000
    if len(data) > threshold:
        to_keep_boundaries = np.sort(data.lid)[[num_to_keep,-num_to_keep]]
        to_keep = data[
            (data['lid'] <= to_keep_boundaries[0]) &\
            (data['lid'] >= to_keep_boundaries[1])
        ]
        sample_size = threshold
        print("The dataset is too big (",len(data)," points), sampling", sample_size, "points")
        print("Keeping all points out of the boundaries:", to_keep_boundaries)
        to_sample = data[
            (data['lid'] > to_keep_boundaries[0]) &\
            (data['lid'] < to_keep_boundaries[1])
        ]
        weight = len(to_sample) / sample_size
        sampled = to_sample.sample(sample_size)
        sampled['weight'] = weight
        to_keep['weight'] = 1
        data = pd.concat([to_keep, sampled])
        geom_density_mapping = aes(weight='weight')
        rugdata = pd.concat([
            sampled[sampled['lid'] < percentiles[2]],
            data[data['lid'] >= percentiles[2]]
        ])

    g = (ggplot(data, aes(x='lid'))
         + geom_density(mapping=geom_density_mapping,
                        color='red', fill='red', alpha=0.6)
         + geom_rug(data=rugdata)
         + geom_vline(xintercept=percentiles[0], linetype='dashed')
         + geom_vline(xintercept=percentiles[1], color='black')
         + geom_vline(xintercept=percentiles[2], linetype='dashed')
         + xlab('lid')
         + theme_bw()
         + theme(figure_size=(10,3)))

    g.draw()
    g.save('{}.png'.format(path), dpi=300)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python3 -m ann_benchmarks.plotting.lid LID_FILE")
        sys.exit()
    lid(sys.argv[1])
