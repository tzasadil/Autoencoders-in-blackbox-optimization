import os
import webbrowser
import cocoex
import cocopp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
matplotlib.use('TkAgg')
from scipy import stats

# computes relative order instead of absolute measurents => model performance can now be compared across different functions and instances
def compute_ranks(df, eval_limit=None):
        
        def to_ranks2(arr):
            arr = arr.to_list()
            dataset = np.array(arr)
            res = []
            for i in range(min(dataset.shape[1], (eval_limit if eval_limit is not None else 9999999))):
                col = dataset[:,i]
                bigger = (col[:,None]>col[None,:]).sum(axis=1)
                eq = (col[:,None]==col[None,:]).sum(axis=1) -1
                percentiles =100-(bigger + (eq/2))/(len(col))*100
                res.append(percentiles)
            r = np.array(res).T
            return list(r)

        def to_ranks(arr):
            arr = arr.to_list()
            res = [[] for _ in range(len(arr))]
            for i in range(max(map(len, arr))):
                a = [(aa[i] if i < len(aa) else None) for aa in arr]
                nonempty_mask = [aa is not None for aa in a]
                nonempty = np.array([aa for aa in a if aa is not None])
                if len(nonempty)==0:break
                if len(nonempty)==1:
                    i = np.argmax(nonempty_mask)
                    res[i].append(100)
                    continue

                bigger = (nonempty[:,None]>nonempty[None,:]).sum(axis=1)
                eq = (nonempty[:,None]==nonempty[None,:]).sum(axis=1) -1
                percentiles =100-(bigger + (eq/2))/(len(nonempty))*100

                # iaia = 100-(nonempty.argsort().argsort())*100/(len(nonempty))  # rank within each column and turn to percentils
                # ieaie = [stats.percentileofscore(-nonempty,-ii) for ii in nonempty]
                
                p_i = 0
                for i,m in enumerate(nonempty_mask):
                    if m == True:
                        res[i].append(percentiles[p_i])
                        p_i+=1
                assert(p_i == len(percentiles))

            # b = np.apply_along_axis(lambda a: (a-a.min())/(a.max()-a.min()),0,a) # normalised to 0-1
            return [np.array(r) for r in res]
        
        def normalise_val_length(df:pd.DataFrame):
            if eval_limit is not None:
                only_evals = df.apply(lambda r: r['evals'][:eval_limit],axis=1)
            else:
                only_evals = df['evals']


            
            # all_steps = [e[1] - e[0] for e in only_evals] 
            # master_step = min(all_steps)
            # max_eval = max([e[-1] for e in only_evals])
            
            # master_evals = np.array(range(5,max_eval+1,master_step))
            
            def vals_to_correct_sampling(run):
                evals = run['evals']
                vals = run['vals']
                dim = run['dim']
                sug_pop =  dim*5
                if eval_limit is not None:
                    evals = evals[:eval_limit]
                    vals = vals[:eval_limit]
                

                if sug_pop == run['pop_size']: return vals
                sug_eval = np.array(range(sug_pop,250*dim+1,sug_pop))

                cur_step = evals[0]
                # aaa =  np.argmin(evals>=master_evals[0])
                begin_i = np.argmin(evals>=sug_pop) #throw away too low
                evals,vals = evals[begin_i:],vals[begin_i:]
                k = int(cur_step/sug_pop) # how many times is cur_step bigger than master_step
                
                
                res = []        
                cur_i = 0
                for master_eval in sug_eval:
                    while cur_i < len(evals) and master_eval > evals[cur_i]:
                        cur_i+=1
                    if cur_i >= len(evals):  #past end of arr
                        res.append(vals[-1])
                    else:       
                        res.append(vals[cur_i])
                return np.array(res)

            transformed = df.apply(vals_to_correct_sampling,axis=1)
            # ahe = transformed.value_counts().to_numpy()
            # print(ahe)
            # lens = [len(a) for a in transformed]
            # u,inx, cs = np.unique(lens,return_index=True,return_counts=True )
            # aaaa = df.loc[inx[0]].to_list()
            # end_i = min(lens)
            # transformed = [a[:end_i] for a in transformed] #all evals should end the same
            # master_evals = master_evals[:end_i]
            df['normalised_len_vals'] = transformed

            return df
        
        df = normalise_val_length(df)
        df['ranks'] = df.groupby(['function', 'instance', 'dim'])['normalised_len_vals'].transform(to_ranks2)
        df = df.drop(['normalised_len_vals'], axis=1)
        df['avg_rank'] = df['ranks'].apply(np.mean) 
        df['last_rank'] = df['ranks'].apply(lambda a: a[-1]) 
        df['median_rank'] = df['ranks'].apply(np.median) 
        return df 
    
def plot(df,window_size = 5):
    df,common_eval = compute_ranks(df)
    plot_ranks(df,common_eval,window_size)

def get_param_desc_title(df):
    title_stringer = lambda beginning, name: beginning + ' ' + str(df[name].min()) + (('-'+str(df[name].max())) if df[name].min() != df[name].max() else '')
    title = title_stringer('fun','function') + '; dim ' + ', '.join([str(a) for a in np.unique(df['dim'])]) + title_stringer('; inst','instance')
    return title

def plot_ranks(df, window_size = 5):
    df = df.copy()
    df['evals'] = df.apply(lambda a: a['evals']/a['dim'], axis=1)
    fig, ax = plt.subplots()
    tit = 'ranks\n'+get_param_desc_title(df)
    ax.set(
        xlabel='evals', 
        ylabel='Rank Percentile',
        title= tit,
        xscale = 'linear',
        yscale='linear'
    )
    # ax.invert_yaxis()  # better ranks to be upper instead of lower on the graph
    ax.grid()
    chunkyfy = lambda arr,window_size: np.array_split(arr, math.ceil(len(arr)/window_size))

    avg_rank_series = df.groupby(['surrogate'])['ranks'].apply(lambda a:np.average(a.to_list(),axis=0)) # rank achieved by each setting in time, avg across fun&dim
    eval_checkpoints = list(map(lambda a: a[-1],chunkyfy(df['evals'].iloc[0],window_size)))
    for (desc, ranks) in avg_rank_series.items():
        # ax.plot(common_eval, ranks,label=desc, linestyle='-', marker='|')
        r = list(map(np.average,chunkyfy(ranks,window_size)))
        ax.plot(eval_checkpoints,r,label=desc, linestyle='-', marker='.')

    #sort legend  
    def sort_legend(ax, values):
        order = np.argsort(values)[::-1]
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(values[idx],2)) for idx in order])
    
    avg_rank = avg_rank_series.apply(np.average) # avg across time from avg across fun&dim, 'final score' of a setting
    sort_legend(ax, avg_rank)
    
    n = datetime.datetime.now().strftime("%m_%d___%H_%M_%S")
    tit = tit.replace('\n', '')
    fig.savefig(f"graphs/ranks_{tit}.png")
    plt.pause(0.01)
    plt.show()
    
# def boxplot():
#   fig, ax = plt.subplots()
#     ax.set(xlabel='log evals', ylabel='func value',
#     title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')
        
    
#     fun_mins = []
#     for config,config_res in zip(configs,results):
    
#         config_desc = config[-1]
#         evals, vals_ = list(map(np.array,zip(*config_res)))
#         # ii = np.argmax(evals[0]>100)
#         # evals = evals[:,ii:]
#         # vals_ = vals_[:,ii:]
#         for i,med in enumerate(vals_):
#             # med = np.median(vals,0)
#             ax.plot(evals[i], med,label=config_desc)
#             fun_mins.append(med[-1])
#             # print([a[-1] for a in med])
#     order = np.argsort(fun_mins)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(fun_mins[idx],2)) for idx in order])
#     ax.grid()
#     graphs = os.listdir('graphs')
#     ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
#     fig.savefig(f"graphs/{ord}.png")
#     # plt.show()

def coco_plot(out_folders):
    
    ### coco plotting 
    # cocopp.genericsettings.background = {None: cocopp.bbob.get_all("2009/")}
    # npz data may store Windows-style paths (exdata\\algo); Linux needs os.sep.
    norm = [
        os.path.normpath(str(p).replace("\\", os.sep))
        for p in out_folders
        if not pd.isna(p)
    ]
    cocopp.genericsettings.isExpensive = True
    cocopp.genericsettings.xlimit_expensive = 250.0
    cocopp.genericsettings.isConv = True
    cocopp.main(" ".join(norm))
    # cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")