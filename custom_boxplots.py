import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from matplotlib.cbook import _reshape_2D

def my_boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                     autorange=False, percents=[25, 75], wis=True):

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:
            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # median
        med = np.percentile(x, 50)
        ## Altered line
        q1, q3 = np.percentile(x, (percents[0], percents[1]))

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        elif not wis:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        elif not wis:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats


def draw_boxplots(perf, names=None, percents=[25,75], scatter=False, 
                  wis=True, path=None, dict_props=None, plot_format="png"):
    
    color = plt.cm.Set3(np.linspace(0, 1, len(perf)))
    
    b_stats, scatter, xs = [], [], []
    stats = {}
    if names is None:
        names = np.arange(1, len(perf)+1, 1)

    for i, group in enumerate(perf):
        stats[names[i]] = my_boxplot_stats(np.array(group), labels=[names[i]], percents=percents, wis=wis)[0]
        b_stats.append(stats[names[i]])
        scatter.append(group)
        xs.append(np.random.normal(i+1, 0.04, group.shape[0]))


    fig, ax = plt.subplots(1,1, figsize=(14,6))

    if scatter:
        for x, val, cc in zip(xs, scatter, color):
            ax.scatter(x, val, color="black", alpha=1, s=12)

    boxes = ax.bxp(b_stats, showmeans=True, meanline=True,
                   vert=True, patch_artist=True, showfliers=False, 
                   shownotches=False, manage_ticks=False, 
                   meanprops={'color':'red', 'linewidth':2}, 
                   medianprops={'color':'purple', 'linewidth':2, 'ls':':'})

    for patch, color in zip(boxes['boxes'], color):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_zorder(0)
    
    plt.legend([boxes['medians'][0], boxes['means'][0]], ['median', 'mean'], fontsize=12)
    if dict_props:
        try:
            plt.title(dict_props["title"], fontdict={"fontsize":18})
            plt.ylabel(dict_props["ylabel"], fontdict={"fontsize":18})
            plt.xticks(np.arange(1, len(names)+1), names, rotation=40, fontdict={"fontsize":14})
        except KeyError:
            pass

    if path:
        if plot_format == 'pgf':
            import texfig                
            texfig.savefig(os.path.join(path, f"boxplot"), bbox_inches='tight')
            return b_stats
        
        plt.savefig(os.path.join(path, f"boxplot.{plot_format}"), bbox_inches='tight', dpi=250)
    plt.show()

    return b_stats