# Draft of representing uncertainty in plots. This script needs a file named
# results.csv in the same directory. Such a file can be obtained with the
# command
#
#     python3 to_csv.py --output results.csv


import numpy as np
from plotnine import *
import pandas as pd
import re

def do_plot(data, output, lines=False, uncertainty=None):
    dataset = set(data['dataset'].values)
    assert len(dataset) == 1
    dataset = list(dataset)[0]
    print(data.columns)
    data['spq'] = 1/data['qps']
    data['spq-dev'] = 1/data['qps-dev']
    data['spq_min'] = data['spq'] - data['spq-dev']
    data['spq_max'] = data['spq'] + data['spq-dev']
    data['knn_min'] = data['k-nn'] - data['k-nn-std']
    data['knn_max'] = data['k-nn'] + data['k-nn-std']
    data['knn'] = data['k-nn']
    print(data[['parameters', 'group_param', 'knn', 'k-nn-std', 'spq', 'spq-dev']])

    g = ggplot(data, aes(x='knn', y='spq', 
                         group='group_param',
                         color='algorithm', fill='algorithm'))
    if uncertainty == 'bars':
        g = (g + geom_errorbar(aes(ymax='spq_max',
                                  ymin='spq_min'),
                              width=0.0)
             + geom_errorbarh(aes(xmax='knn_max',
                                  xmin='knn_min')))
    elif uncertainty == 'rect':
        g = (g + geom_rect(aes(xmax='knn_max',
                               xmin='knn_min',
                               ymax='spq_max',
                               ymin='spq_min'),
                           color=None,
                           alpha=0.1))
    g = g + geom_point(size=0.1)
    if lines:
         g = g + geom_line()

    g = (g + scale_y_log10()
         # + facet_grid(('dataset', 'algorithm'), scales='fixed')
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
    data = pd.read_csv('results.csv')
    data['group_param'] = [get_grouping_parameter(p) for p in data['parameters']]
    datasets = set(data['dataset'].values)
    for dataset in datasets:
        plotdata = data[data['dataset'] == dataset]
        do_plot(plotdata, 'uncertainty/{}.png'.format(dataset), uncertainty='rect')
