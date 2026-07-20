import numpy as np
from cmaes import CMA
import cma
# from cma.purecma import CMAES
# from cma_custom import CMA
#cocoex.solvers.random_search
import progress_bar
from scipy import stats
import cocoex


def optimize(problem, surrogate, pop_size, true_evals, gen_mult:int, printing=True,seed = 42):
    rng = np.random.default_rng()
    cloned_problem = cocoex.Suite("bbob", '', '').get_problem_by_function_dimension_instance(*problem.id_triple)
    optimizer= None
    dim=problem.dimension
    # bounds = np.stack([problem.lower_bounds,problem.upper_bounds],axis=0).T
    dist = -1
    optimizer_popsize = pop_size
    def new_optim(dim=dim, optimizer_popsize=optimizer_popsize):
        # initial = np.random.rand(problem.dimension)*9 - 4.5
        initial = np.array([0.1]*dim)
        return CMA(mean=initial, sigma=1.0, seed=seed,bounds=np.array([[-5.0,5.0]]*dim),population_size=optimizer_popsize)
    next_specimens_forced = []
    def next_gen(size=optimizer_popsize):
        nonlocal next_specimens_forced
        forced = next_specimens_forced[:size]
        next_specimens_forced = next_specimens_forced[size:]
        selected = []
        seen = set()

        def add_if_new(x):
            key = np.ascontiguousarray(x).tobytes()
            if key in seen:
                return False
            seen.add(key)
            selected.append(np.array(x, copy=True))
            return True

        for x in forced:
            add_if_new(x)

        max_attempts = max(10 * size, size + 10)
        attempts = 0
        while len(selected) < size and attempts < max_attempts:
            add_if_new(optimizer.ask())
            attempts += 1

        # fill remaining slots even if ask() keeps returning duplicates
        while len(selected) < size:
            selected.append(np.array(optimizer.ask(), copy=True))

        return np.array(selected)

    # if warm_start_task != None:
    #     source_solutions = []
    #     for _ in range(1000):
    #         x = np.random.random(warm_start_task.dimension)
    #         value = warm_start_task(x)
    #         source_solutions.append((x, value))
    #     # ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(true_points, gamma=0.5, alpha=0.1)
    #     # optimizer = CMA(mean=ws_mean, sigma=ws_sigma,cov=ws_cov,bounds=bounds,population_size=pop_size)
    # else:
       # should there be popsize or K???
    current_model_uses = 0
    evals_wihout_change = 0
    true_xs= []
    true_ys= []
    true_evals_left = true_evals
    best = 9999999999
    best_x = np.zeros(problem.dimension)
    overall_best = 9999999999
    overall_best_x = 9999999999
    bests,bests_evals = [],[] #best found values overall and timestamps of currently used evaluations
    spearman_corr = []
    spearman_pval = []
    corr_invariant_tracker = []
    selected_spread_ratio = []
    selected_radius_ratio = []
    selection_quality_gap = []
    oracle_regret = []

    def mean_pairwise_distance(points):
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return np.nan
        diffs = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        upper = distances[np.triu_indices(len(points), k=1)]
        return float(np.mean(upper)) if upper.size else np.nan

    def mean_radius(points, center):
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return np.nan
        center = np.asarray(center, dtype=float).reshape(1, -1)
        return float(np.mean(np.linalg.norm(points - center, axis=1)))

    def cluster_selector_indices(xs, correct_ys, k_accepted, mode, n_clusters, selector_seed):
        from sklearn.cluster import KMeans

        xs = np.asarray(xs, dtype=float)
        correct_ys = np.asarray(correct_ys, dtype=float)
        sample_count = len(xs)
        if sample_count == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        cluster_count = max(1, min(int(n_clusters), sample_count))
        if cluster_count == 1:
            rank_scores = np.ones(sample_count, dtype=float)
            if mode == "cluster_best_half_oracle":
                order = np.argsort(correct_ys)
                rank_scores += correct_ys / (np.nanmax(np.abs(correct_ys)) + 1e-12)
            else:
                order = np.random.default_rng(selector_seed).permutation(sample_count)
                rank_scores += np.random.default_rng(selector_seed).random(sample_count) * 1e-6
            rank_scores[order[:k_accepted]] = 0.0
            return order[:k_accepted], rank_scores

        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=selector_seed)
        labels = kmeans.fit_predict(xs)
        rng_local = np.random.default_rng(selector_seed)
        cluster_ids = np.arange(cluster_count, dtype=int)

        if mode == "cluster_best_half_oracle":
            cluster_scores = np.array(
                [np.mean(correct_ys[labels == cluster_id]) for cluster_id in cluster_ids],
                dtype=float,
            )
            ordered_clusters = cluster_ids[np.argsort(cluster_scores)]
        else:
            ordered_clusters = rng_local.permutation(cluster_ids)

        selected_indices = []
        for cluster_id in ordered_clusters:
            cluster_idx = np.flatnonzero(labels == cluster_id)
            if mode == "cluster_best_half_oracle":
                cluster_idx = cluster_idx[np.argsort(correct_ys[cluster_idx])]
            else:
                cluster_idx = rng_local.permutation(cluster_idx)
            remaining = k_accepted - len(selected_indices)
            if remaining <= 0:
                break
            selected_indices.extend(cluster_idx[:remaining].tolist())

        if len(selected_indices) < k_accepted:
            remaining_idx = np.setdiff1d(np.arange(sample_count), np.array(selected_indices, dtype=int), assume_unique=False)
            if mode == "cluster_best_half_oracle":
                remaining_idx = remaining_idx[np.argsort(correct_ys[remaining_idx])]
            else:
                remaining_idx = rng_local.permutation(remaining_idx)
            selected_indices.extend(remaining_idx[: k_accepted - len(selected_indices)].tolist())

        selected_indices = np.array(selected_indices[:k_accepted], dtype=int)
        rank_scores = np.ones(sample_count, dtype=float)
        rank_scores[selected_indices] = 0.0
        if mode == "cluster_best_half_oracle":
            rank_scores += correct_ys / (np.nanmax(np.abs(correct_ys)) + 1e-12)
        else:
            rank_scores += rng_local.random(sample_count) * 1e-6
        return selected_indices, rank_scores

    def eval_true(xs):
        nonlocal true_evals_left,true_evals,bests,bests_evals,printing,best,problem,true_xs,true_ys,evals_wihout_change,optimizer,overall_best,best_x,overall_best_x
        ys = np.array([problem(x) for x in xs])
        ys = np.where(np.isinf(ys), 1e11, ys)
        ys = np.where(np.isnan(ys), 1e11, ys)

        true_xs += list(xs)
        true_ys += list(ys)
        true_evals_left -= xs.shape[0]
        if true_evals_left < 0:
            print()
        if best > np.min(ys):
            index = np.argmin(ys)
            best = ys[index]
            best_x = xs[index]
            if overall_best > best:
                overall_best = best
                overall_best_x = best_x
            evals_wihout_change = 0
        else:
            evals_wihout_change += xs.shape[0]
            if evals_wihout_change > 2000:
                optimizer = new_optim()
                best = 9999999999
        bests.append(overall_best)
        bests_evals.append(true_evals-true_evals_left)
        if printing:
            progress_bar.progress_bar(overall_best,true_evals-true_evals_left,true_evals)
        if printing and true_evals_left <= 0:
            print(' '*80,end='\r') #deletes progress bar
        return ys

    generation = 0
    # mean_weights = []
    if False:
        es = cma.CMAEvolutionStrategy ( problem.dimension * [0.1], 0.1 )
        surrogate = cma.fitness_models.SurrogatePopulation(problem)
        while not es.stop():
            X = es.ask() # sample a new population
            F = surrogate( X ) # see Algorithm 1
            es.tell(X , F ) # update sample distribution
            es.inject([ surrogate.model.xopt ])
            es.disp() # just checking what 's going one
        return es.best.f
    if True or gen_mult !=1:
        generated_population = int(pop_size * gen_mult)
        optimizer_popsize = pop_size
        if optimizer is None:
            optimizer = new_optim(optimizer_popsize=optimizer_popsize)

        while True:
            xs = next_gen(optimizer_popsize)
            ys = eval_true(xs)
            optimizer.tell(list(zip(xs,ys)))
            if len (true_ys) >= surrogate.inp_size or true_evals_left <= 0:
                break
        surrogate.train(true_xs,true_ys, opt=optimizer)
        xs = next_gen(generated_population)
        while true_evals_left > 0:
            xs = next_gen(generated_population)
            xs = np.array(xs)
            correct_ys = np.array([cloned_problem(x) for x in xs])
            k_accepted = min(true_evals_left,optimizer_popsize)
            selection_mode = getattr(surrogate, "selection_mode", "")
            if selection_mode == "oracle":
                ys = correct_ys.copy()
            elif selection_mode == "negative_oracle":
                ys = -correct_ys.copy()
            elif selection_mode in {"cluster_random_half", "cluster_best_half_oracle"}:
                selected_idx, ys = cluster_selector_indices(
                    xs,
                    correct_ys,
                    k_accepted,
                    selection_mode,
                    getattr(surrogate, "n_clusters", 4),
                    getattr(surrogate, "seed", seed) + generation,
                )
                idx = selected_idx
            else:
                ys = surrogate(xs)
            ys = np.array(ys)
            if selection_mode in {"cluster_random_half", "cluster_best_half_oracle"}:
                selected_mask = np.zeros(len(xs), dtype=bool)
                selected_mask[idx] = True
                rejected_idx = np.flatnonzero(~selected_mask)
                ordering = np.concatenate([idx, rejected_idx])
            else:
                ordering = np.argsort(ys)
                idx = ordering[:k_accepted]
            top_k_xs = xs[idx][:k_accepted]
            top_k_true_from_pool = correct_ys[idx][:k_accepted]
            rejected_true_from_pool = correct_ys[ordering][k_accepted:]
            oracle_best_true = np.sort(correct_ys)[:k_accepted]
            top_k_ys = eval_true(top_k_xs)
            if true_evals_left >0: # at the end of algo; optim complains the solutions have diff len than popsize; wont continue, so no need to tell optimizer anyway
                optimizer.tell(list(zip(top_k_xs,top_k_ys)))
                surrogate.train(true_xs,true_ys, opt=optimizer)


            # avg_err = np.average(np.abs(top_k_ys - ys[:k]))
            generation += 1

            #corr stat computation
            sp = stats.spearmanr(correct_ys, ys)
            spearman_corr.append(sp.correlation)
            spearman_pval.append(sp.pvalue)
            pool_spread = mean_pairwise_distance(xs)
            chosen_spread = mean_pairwise_distance(top_k_xs)
            if np.isfinite(pool_spread) and pool_spread > 1e-12 and np.isfinite(chosen_spread):
                selected_spread_ratio.append(chosen_spread / pool_spread)
            else:
                selected_spread_ratio.append(np.nan)
            center = optimizer._mean
            pool_radius = mean_radius(xs, center)
            chosen_radius = mean_radius(top_k_xs, center)
            if np.isfinite(pool_radius) and pool_radius > 1e-12 and np.isfinite(chosen_radius):
                selected_radius_ratio.append(chosen_radius / pool_radius)
            else:
                selected_radius_ratio.append(np.nan)
            if len(rejected_true_from_pool) > 0:
                selection_quality_gap.append(float(np.mean(rejected_true_from_pool) - np.mean(top_k_true_from_pool)))
            else:
                selection_quality_gap.append(np.nan)
            oracle_regret.append(float(np.mean(top_k_true_from_pool) - np.mean(oracle_best_true)))

    # if isinstance(gen_mult, Pure):
    #     if optimizer == None:
    #         optimizer = new_optim(optimizer_popsize=pop_size)
    #     while true_evals_left > 0 :
    #         xs = next_gen()
    #         if xs.shape[0] > true_evals_left:
    #             xs = xs[:true_evals_left]
    #         ys = eval_true(xs)
    #         if true_evals_left > 0:
    #             optimizer.tell(list(zip(xs,ys)))
    #         generation += 1


    def plotty():
        import matplotlib.pyplot as plt
        import scipy.stats
        plt.scatter(dists, corr_invariant_tracker)
        m, b = np.polyfit(dists, corr_invariant_tracker, 1)
        lr = scipy.stats.linregress(dists, corr_invariant_tracker)
        xx = np.linspace(0, np.max(dists), num=100)
        plt.plot(xx, lr.slope*xx + lr.intercept,color='red', alpha=0.5)
        plt.annotate("r-squared = {:.3f}".format(lr.rvalue**2), (0, 1))
        plt.show()


    if hasattr(surrogate, 'distances'):
        dists = surrogate.distances
        # import matplotlib.pyplot as plt
        # import scipy.stats
        # plt.scatter(dists, corr_invariant_tracker)
        # m, b = np.polyfit(dists, corr_invariant_tracker, 1)
        # lr = scipy.stats.linregress(dists, corr_invariant_tracker)
        # xx = np.linspace(0, np.max(dists), num=100)
        # plt.plot(xx, lr.slope*xx + lr.intercept,color='red', alpha=0.5)
        # plt.annotate("r-squared = {:.3f}".format(lr.rvalue**2), (0, 1))
        # plt.show()
    else:
        dists = np.zeros(len(spearman_corr))
    return (
        np.array(bests_evals),
        np.array(bests),
        np.array(spearman_corr),
        np.array(spearman_pval),
        np.array(dists),
        np.array(selected_spread_ratio),
        np.array(selected_radius_ratio),
        np.array(selection_quality_gap),
        np.array(oracle_regret),
    )














if __name__ == '__main__':
    import main
    main.main()
