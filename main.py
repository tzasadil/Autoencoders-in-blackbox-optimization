#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""

from __future__ import division, print_function
import cocopp.preparetexforhtml
import cocopp.testbedsettings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

threadpool = ProcessPoolExecutor(max_workers=16)
import cocoex.function
import cocoex, cocopp  # experimentation and post-processing modules
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
import evo
from timeit import default_timer as timer
import pandas as pd
import sklearn.gaussian_process.kernels as GPK
from datetime import datetime
from functools import partial as p
import matplotlib
import models
import ranks
import storage
import pd_cols
from doe2vec.doe2vec import doe_model
import math
import functools
from itertools import takewhile, dropwhile

matplotlib.use("TkAgg")
os.environ["KERAS_BACKEND"] = "tensorflow"

if not os.path.exists("./doe_saves"):
    os.makedirs("./doe_saves")
if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./exdata"):
    os.makedirs("./exdata")
    # tf.config.run_functions_eagerly(True)
    # tf.keras.backend.set_floatx('float16')

#
# num of sampling points
# latent size
# knn > 1 as ansamble?
# diff similarities
#
# sampling -5 5 instead of 0 1
#
#
# count the sample eval as true eval or not
#
# make x_min_len scale with dim
#
#
#
#
budget = int(250)  # times dim
rrr = cocopp.testbedsettings.current_testbed

listmap = lambda func, collection: list(map(func, collection))

unzip = lambda a: list(map(list, list(zip(*a))))
func_names = [
    "Sphere",
    "Ellipsoidal",
    "Rastrigin",
    "B ̈uche-Rastrigin",
    "Linear Slope",
    "Attractive Sector",
    "Step Ellipsoidal",
    "Rosenbrock",
    "Rosenbrock rotated",
    "Ellipsoidal",
    "Discus",
    "Bent Cigar",
    "Sharp Ridge",
    "Different Powers",
    "Rastrigin",
    "Weierstrass",
    "Schaffers F7",
    "Schaffers F7 moderately ill-conditioned",
    "Composite Griewank-Rosenbrock F8F2",
    "Schwefel",
    "Gallagher’s Gaussian 101-me Peaks",
    "Gallagher’s Gaussian 21-hi Peaks",
    "Katsuura",
    "Lunacek bi-Rastrigin",
]
cached_doe_functions = None


def main(df=None):
    if df is None:
        df = storage.load_data()
    df = run(df)
    plot(df)


def plot(df=None):
    if df is None:
        df = storage.load_data()
    ranks.plot(df)


note = ""


def run(df=None):
    # global dim, budget
    problem_info = f"function_indices:1-24 dimensions:2,5,10 instance_indices:1-10"

    vae = lambda layers: (p(models.vae, layers), f"vae{layers}")
    pca = lambda n: (p(models.pca, n), f"pca{n}")

    elm = lambda nodes: (p(models.elm, nodes), f"elm{nodes}")
    rbf = lambda layers, gamma: (
        p(models.rbf_network, layers, gamma),
        f"rbf{layers}_{gamma}",
    )
    gp = p(models.gp, GPK.Matern(nu=5 / 2)), "gp"
    nearest = lambda k: (p(models.nearest, k), f"nn{k}")
    mlp = lambda nodes: (p(models.mlp, nodes), f"mlp{nodes}")
    doe = lambda n_samples, latent: (d := doe_model(n_samples, latent), str(d))

    # ans1 =models.ansamble.create(np.median,[gp, elm(200), rbf([1/2], 1)])

    configs = [
        ## pop_size, evolution_eval_mode, dim_reduction, model,budget, train_num, sort_train, scale_train, cma_sees_approximations
        [None, 4, None, doe(16, 4)],
        [None, 1, None, None],
        [None, 4, None, gp],
        [None, 4, None, nearest(3)],
        [None, 4, None, elm(100)],
        # [None, 4, None, nearest(1)],
        # [None, 4, None, nearest(2)],
        #     [None, 2,None, doe(32, 4)],
        #     [None, 2,None, doe(32, 16)],
        #     [None, 2,None, doe(32, 32)],
    ]

    # for mult in [2,4,6,8,12,16]:
    #     for pop in [2,4,6,8,12,16]:
    #         configs.append([pop*mult,best_k(1.0/mult),None,gp, budget, 200, False, False, False])

    # for model in [rbf([1], 10.0),rbf([5], 5.0),elm(10),elm(30), mlp([100]),mlp([1,1,1]),mlp([10,10,10]), mlp([10,10])]:
    #     for pop in [48]:
    #         configs.append([pop,best_k(1.0/8),None,model,budget, 200, False, False,False])

    for config in configs:
        res = single_config(config, problem_info, df=df)
        df = pd.concat([df, res], ignore_index=True)

    evo.last_cached_doe = None
    return df


def single_config(config, problem_info, df=None):
    global cached_doe_functions, budget
    pop_size, gen_mult, dim_red, model = config
    pop_size = int(pop_size) if pop_size is not None else pop_size
    (dim_red_f, dim_red_name) = dim_red if dim_red else (None, "")
    (model_f, model_name) = model if model else (None, "")
    pop_size = pop_size if pop_size is not None else "None"
    full_desc = (
        f"{pop_size}_{gen_mult}"
        + ("_" if len(dim_red_name) > 0 else "")
        + f"{dim_red_name}"
        + ("_" if len(model_name) > 0 else "")
        + f"{model_name}_"
        + f"{note}"
    )
    opts = {
        "algorithm_name": "".join(takewhile(lambda s: s.isalnum(), model_name))
        if len(model_name) > 0
        else "no surrogate",
        "algorithm_info": '"autoencoder_surrogate"',
        "result_folder": "".join(takewhile(lambda s: s.isalnum(), model_name))
        if len(model_name) > 0
        else "no surrogate",
    }

    suite = cocoex.Suite("bbob", "", problem_info)
    minimal_print = cocoex.utilities.MiniPrint()
    observer = cocoex.Observer("bbob", opts)
    df_filtered = df

    # filter already done experiments to find those same as this one
    if df_filtered is not None:
        vals = [pop_size, gen_mult, model_name, dim_red_name, budget, note]
        assert len(pd_cols.determining_cols) == len(vals)
        df_filtered = df
        for n, v in zip(pd_cols.determining_cols, vals):
            df_filtered = df_filtered[df_filtered[n] == v]

    results = []
    for problem in suite:
        fun, dim, ins = problem.id_triple
        # check if this exact experiment was already run
        if df_filtered is not None:
            names = ["instance", "function", "dim"]
            vals = [ins, fun, dim]
            df_filtered_local = df_filtered
            for n, v in zip(names, vals):
                df_filtered_local = df_filtered_local[df_filtered_local[n] == v]
            # masks = np.array([(df[n] == v).to_numpy() for n,v in zip(names,vals)]).T
            # is_run_duplicate = np.logical_and.reduce(masks,axis=1)
            # is_run_duplicate = np.any(is_run_duplicate)
            if not df_filtered_local.empty:  # check if this run is duplicate:
                continue

        if isinstance(model_f, doe_model):
            model_f.functions = cached_doe_functions
            surrogate = model_f.load_or_create(dim)
            cached_doe_functions = model_f.functions
            surrogate.executor = threadpool
        else:
            surrogate = models.Surrogate(model_f, dim_red_f)
        observer.observe(problem)
        start_time = timer()
        pop_none = 5 * dim
        evals, vals, spearman_corr, spearman_pval, dists = evo.optimize(
            problem,
            surrogate,
            # pop_size = 4 + math.floor(3 * math.log(dim)) if pop_size == 'None' else int(pop_size) if isinstance(pop_size, int) else float(pop_size) * (4 + math.floor(3 * math.log(dim))),
            pop_size=pop_none
            if pop_size == "None"
            else int(pop_size)
            if isinstance(pop_size, int)
            else int(float(pop_size) * pop_none),
            true_evals=budget * dim,
            printing=True,
            seed=42,
            gen_mult=gen_mult,
        )
        end_time = timer()
        problem.free()
        elapsed = end_time - start_time
        print(f"single setting time: {elapsed}")
        timestamp = datetime.now().strftime("%m_%d___%H_%M_%S")

        df_row = [
            vals,
            evals,
            pop_size,
            gen_mult,
            model_name,
            dim_red_name,
            ins,
            fun,
            dim,
            elapsed,
            observer.result_folder,
            timestamp,
            budget,
            note if gen_mult != 1 else "",
            spearman_corr,
            spearman_pval,
            dists,
        ]
        ddff = pd.DataFrame({k: v for (k, v) in zip(pd_cols.all_cols, unzip([df_row]))})
        dsc = pd_cols.get_storage_desc(ddff)
        storage.store_data(ddff, dsc)
        results.append(df_row)

    if isinstance(model_f, doe_model):
        cached_doe_functions = model_f.functions

    print(
        f"....................................................................run complete {config}"
    )

    res = pd.DataFrame(
        {k: v for (k, v) in zip(pd_cols.all_cols, unzip(results))}
    )  # converts list of dataframe slices to dataframe
    return res


def coco_gen():
    df_og = storage.merge_and_load()
    # df_og = run(df_og)

    # from doe2vec import exp_bbob

    df = df_og.copy()
    df = df[df["note"] == ""]
    out_folders = df["coco_directory"].unique().tolist()
    ranks.coco_plot(out_folders)


def execute_optimization():
    df_og = storage.merge_and_load()
    df_og = run(df_og)
    plot(df_og)
    # datastore_store(load_data(),'w')


if __name__ == "__main__":
    # ['pure', 'gp', 'elm', rbf]
    # df = storage.merge_and_load()
    # df_og = storage.merge_and_load()
    # df = run(df)
    # plot(df)
    # datastore_store(load_data(),'w')

    df_og = storage.merge_and_load()
    # df_og = run(df_og)

    # from doe2vec import exp_bbob

    df = df_og.copy()
    df = df[df["note"] == ""]
    out_folders = df["coco_directory"].unique().tolist()
    ranks.coco_plot(out_folders)
