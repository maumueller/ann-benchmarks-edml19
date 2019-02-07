import os
import sys
import matplotlib
# I get a crash without this line
matplotlib.use('Agg')
import numpy as np
from plotnine import *
import pandas as pd
import re

def uncertainty(data, output, x='knn', y='spq', x_dev=None, y_dev=None, lines=False, uncertainty=None):
    if x_dev is None:
        x_dev = x + '-std'
    if y_dev is None:
        y_dev = y + '-std'

    dataset = set(data['dataset'].values)
    assert len(dataset) == 1
    dataset = list(dataset)[0]
    data['y_min'] = data[y] - data[y_dev]
    data['y_max'] = data[y] + data[y_dev]
    data['x_min'] = data[x] - data[x_dev]
    data['x_max'] = data[x] + data[x_dev]

    g = ggplot(data, aes(x=x, y=y, 
                         group='group_param',
                         color='algorithm', fill='algorithm'))
    if uncertainty == 'bars':
        g = (g + geom_errorbar(aes(ymax='y_max',
                                  ymin='y_min'),
                              width=0.0)
             + geom_errorbarh(aes(xmax='x_max',
                                  xmin='x_min')))
    elif uncertainty == 'rect':
        g = (g + geom_rect(aes(xmax='x_max',
                               xmin='x_min',
                               ymax='y_max',
                               ymin='y_min'),
                           color=None,
                           alpha=0.1))
    g = g + geom_point(size=0.1)
    if lines:
         g = g + geom_line()

    g = (g + scale_y_log10()
         + ggtitle(dataset)
         + ylab('seconds per query')
         + xlab('recall')
         + theme_bw()
         + theme(figure_size=(10,10)))
    g.draw()
    g.save(output, limitsize=False)


FAISS_IFV_PARAM = re.compile(r'FaissIVF\(n_list=(\d+), n_probe=(\d+)\)')
ONNG_NGT_PARAM = re.compile(r'ONNG-NGT\((\d+), (\d+), (\d+), (-?\d+), (\d+.\d+)\)')
FAISS_PARAM = re.compile(r"faiss \({u'efConstruction': (\d+), u'M': (\d+)}\)")
ANNOY_PARAM = re.compile(r"Annoy\(n_trees=(\d+), search_k=(\d+)\)")


def get_grouping_parameter(param_string):
    m = FAISS_IFV_PARAM.search(param_string)
    if m is not None:
        return m.group(1)
    m = ONNG_NGT_PARAM.search(param_string)
    if m is not None:
        return m.group(5)
    m = FAISS_PARAM.search(param_string)
    if m is not None:
        return m.group(2)
    m = ANNOY_PARAM.search(param_string)
    if m is not None:
        return m.group(1)
    print('Could not find param for', param_string)
    return np.nan

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python3 -m ann_benchmarks.plotting.uncertainty RESULTS_FILE")
        sys.exit()
    data = pd.read_csv(sys.argv[1])

    out_dir = 'uncertainty_plots'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    data['group_param'] = [get_grouping_parameter(p) for p in data['parameters']]
    data['spq'] = 1/data['qps']
    data['spq-std'] = 1/data['qps-dev']
    data['spq'] = 1/data['qps']
    data['spq-std'] = 1/data['qps-dev']
    datasets = set(data['dataset'].values)
    for dataset in datasets:
        plotdata = data[data['dataset'] == dataset]
        uncertainty(plotdata, '{}/{}.png'.format(out_dir,dataset), 
                    x='k-nn',
                    uncertainty='rect')
