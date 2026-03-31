import numpy as np
from cmaes import CMA
import cma
# from cma.purecma import CMAES
# from cma_custom import CMA
#cocoex.solvers.random_search
import progress_bar
from scipy import stats

import cocoex
from cma.fitness_models import SurrogatePopulation
from doe2vec.doe2vec import doe_model 
from doe2vec.bbobbenchmarks import instantiate
from timeit import default_timer as timer
from scipy import stats


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

        # Exact duplicates are wasted candidates for preselection, but keep a fallback
        # so candidate generation cannot stall if the sampler collapses numerically.
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
        if optimizer == None: 
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
            ys = surrogate(xs) 
            xs,ys = np.array(xs), np.array(ys)
            k_accepted = min(true_evals_left,optimizer_popsize)
            idx = np.argsort(ys)
            top_k_xs = xs[idx][:k_accepted]  
            top_k_ys = eval_true(top_k_xs)
            if true_evals_left >0: # at the end of algo; optim complains the solutions have diff len than popsize; wont continue, so no need to tell optimizer anyway
                optimizer.tell(list(zip(top_k_xs,top_k_ys))) 
                surrogate.train(true_xs,true_ys, opt=optimizer)
                

            # avg_err = np.average(np.abs(top_k_ys - ys[:k]))
            generation += 1    

            #corr stat computation
            correct_ys = np.array([cloned_problem(x) for x in xs])
            sp = stats.spearmanr(correct_ys, ys)
            spearman_corr.append(sp.correlation)
            spearman_pval.append(sp.pvalue)

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
    return np.array(bests_evals), np.array(bests), np.array(spearman_corr),np.array(spearman_pval), np.array(dists)



    










if __name__ == '__main__':
    import main
    main.main()