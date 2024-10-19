import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

from utils import check_df_format


def get_ranks(df, asc=True, agg='mean'):
    
    check_df_format(df)

    if agg == "":
        agg = None 

    ranked_array = []
    x_values = df["x_parameter"].unique()
    for x_val in x_values:
        
        df_param = df[df["x_parameter"] == x_val]
        df_param = df_param.sort_values(["dataset", "competitor"])

        labels = sorted(list(df_param["competitor"].unique()))
        
        rank_data = df_param["metric"].values.reshape((-1, len(labels)))
        if agg == 'mean':
            ranked_df_param = pd.DataFrame(data=rank_data, columns=labels) \
                .rank(axis=1, ascending=asc, method='average').mean()
        elif agg == 'median':
            ranked_df_param = pd.DataFrame(data=rank_data, columns=labels) \
                .rank(axis=1, ascending=asc, method='average').median()
        elif agg is None:
            ranked_df_param = pd.DataFrame(data=rank_data, columns=labels)
        else:
            raise ValueError(f"Unknown value {agg}, for 'agg' parameter")

        ranked_array.append(ranked_df_param.values)
    
    try:
        ii = [float(x) for x in x_values]
    except ValueError:
        ii = [str(x) for x in x_values]
    
    if agg is None:
        ii = pd.MultiIndex.from_tuples(
            [(i, d) for i in ii for d in df["dataset"].unique()]
        )

    ranked_df = pd.DataFrame(data=np.array(ranked_array).reshape(-1, len(labels)), index=ii, columns=labels)
    
    return ranked_df

def get_ranks_std(df_flat, confidence_level=0.9):

    stds = pd.DataFrame()    
    for x_val in df_flat.index.levels[0]:

        df_x = df_flat[df_flat.index.get_level_values(0)==x_val]
        df_rank = df_x.rank(axis=1, ascending=True, method="average")
            
        ci = []
        for col in df_rank.columns:
            res = bootstrap((df_rank[col].values,), np.mean, confidence_level=confidence_level, random_state=44)
            ci.append((res.confidence_interval.low, res.confidence_interval.high))

        ciss = pd.DataFrame(data=ci, index=df_rank.columns, columns=[f"low_{x_val}", f"high_{x_val}"])
        stds = pd.concat((stds, ciss), axis=1)

    return stds.T

def bump_chart(df, confidence=True, asc=True, confidence_level=0.9, title=None, xlab="parameter",
               style_dict={}, add_legend=[], save=None, plot_format='pdf'):

    import matplotlib
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.collections import LineCollection

    matplotlib.rcParams.update({'font.size': 18})

    if style_dict is None:
        style_dict = {k: {"marker": "d", "mfc": "w", "alpha":0.9} for k in df['competitor'].unique()}
    
    df_mean_rank = get_ranks(df, asc=asc, agg='mean')
    if confidence:
        df_flat = get_ranks(df, asc=asc, agg=None)
        conf = get_ranks_std(df_flat, confidence_level=confidence_level)

    df = df_mean_rank

    y_min = math.floor(df_mean_rank.min().min())
    y_max = math.ceil(df_mean_rank.max().max())

    kw = dict(xlim=(df.index.min()-2e-2, df.index.max()+2e-2), ylim=(y_min, y_max))

    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw=kw)
    
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.axvline(x=0.5, ls=":", c="gray", lw=3)

    for col in df.columns:
        if conf is not None:
            low = conf[col][conf[col].index.str.contains("low")]
            high = conf[col][conf[col].index.str.contains("high")]
            ax.fill_between(list(df.index), low, high, alpha=0.1, lw=0)
        
        ax.plot(df[col], lw=3, ms=12, label=col, **style_dict[col])

    plt.xticks(df.index)
    ax.invert_yaxis()
    if title:
        ylabel = "mean rank" 
        ax.set(xlabel=xlab, ylabel=ylabel, title=title)
    else:
        ylab = "mean rank"
        ax.set(xlabel=xlab, ylabel=ylab)

    ax.grid(axis="x")
    
    ll = []
    for leg in add_legend:
        leg_add = ax.legend(handles=leg["legend"], **leg["kwargs"])   
        ll.append(leg_add)

    leg1 = ax.legend(loc="lower right", ncols=1, fontsize=16,
                     markerscale=1.2, framealpha=1, bbox_to_anchor=(1.28, 0.2))
    
    for l in ll:
        ax.add_artist(l)

    #args = {"fontstyle": "italic", "fontsize":22, "c":"gray", "weight":"bold"}  
    #plt.text(0.03, 1.75, f"High delay cost", bbox=dict(facecolor='white', edgecolor="gray"), **args)
    #plt.text(0.75, len(df.columns), f"Low delay cost", bbox=dict(facecolor='white', edgecolor='gray'), **args)
           
    plt.tight_layout()
    if save:
        if plot_format == 'pgf':
            import texfig
            texfig.savefig(os.path.join(save, f'bump'), bbox_inches='tight')
        else:
            if title:
                plt.savefig(os.path.join(save, f'bump_{title[0]}.{plot_format}'), bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(save, f'bump.{plot_format}'), bbox_inches='tight', dpi=300)
    else:
        plt.show()