import os.path
import sys
import warnings
from statistics import mode

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
from matplotlib import cm
from numpy.random import seed
from scipy.stats import qmc
from sklearn import manifold
import math
# from doe2vec import bbobbenchmarks as bbob
from doe2vec.vae import VAE
from doe2vec.modulesRandFunc import generate_exp2fun as genExp2fun
from doe2vec.modulesRandFunc import generate_tree as genTree
from doe2vec.modulesRandFunc import generate_tree2exp as genTree2exp
from concurrent.futures import ProcessPoolExecutor

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

def no_descs(ax):
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            for line in axis.get_ticklines():
                line.set_visible(False)
def eval_multiple(fs, array_x):
    fs = [a+' ' for a in fs]
    y = []
    orig_settings = np.seterr(all='raise')
    for f in fs:
        try:
            y.append(eval(f))
        except:
            y.append(None)
    np.seterr(**orig_settings)
    return y
# def evaluator(conn):
#     import numpy as np
#     fs = conn.recv()
#     fs = [eval('lambda array_x:'+f) for f in fs]
#     orig_settings = np.seterr(all='raise')
#     while True:
#         y = []
#         array_x = np.array(conn.recv())
#         for f in fs:
#             try:
#                 y.append(f(array_x))
#             except:
#                 y.append(None)
#         conn.send(y)
#     np.seterr(**orig_settings)
#     return y
def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]

class doe_model:
    def __init__(
        self,
        inp_size,
        latent_dim,
        n_functions=250_000,
        seed_nr=0,
        kl_weight=0.001
    ):
        """Doe2Vec model to transform Design of Experiments to feature vectors.

        Args:
            m (int): Power for number of samples used in the Sobol sampler (not used for custom_sample)
            n (int, optional): Number of generated functions to use a training data. Defaults to 1000.
            latent_dim (int, optional): Number of dimensions in the latent space (vector size). Defaults to 16.
            seed_nr (int, optional): Random seed. Defaults to 0.
            kl_weight (float, optional): Defaults to 0.1.
            custom_sample (array, optional): dim-d Array with a custom sample or None to use Sobol sequences. Defaults to None.
            use_mlflow (bool, optional): To use the mlflow backend to log experiments. Defaults to False.
            mlflow_name (str, optional): The name to log the mlflow experiment. Defaults to "Doc2Vec".
            model_type (str, optional): The model to use, either "AE" or "VAE". Defaults to "VAE".
        """
        self.inp_size_base = inp_size
        self.n_functions = n_functions
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.seed = seed_nr
        self.loaded = False
        self.autoencoder = None
        self.functions = []
        self.active_functions = []
        self.distances = []
        self.fun_save_path = f'doe_saves/functions.npy'
        # self.model_save_path = f'doe_saves/{self.inp_size}_{self.latent_dim}'
        seed(self.seed)
        self.executor:ProcessPoolExecutor =  None
        # worker_n = 8
        # self.worker_conns, child_conns = list(zip(*[Pipe() for _ in range(worker_n)]))
        # self.eval_workers = [Process(target=evaluator, args=(conn,)) for conn in child_conns]
        # for p in self.eval_workers: p.start()


        self.train_epochs = 1
        self.old_xs = None


    def __str__(self):
        return f'doe_{self.inp_size_base}_{self.latent_dim}'

    def _drop_duplicate_points(self, train_x, train_y):
        if len(train_x) == 0:
            return train_x, train_y
        # Repeated evaluations at the same location do not add DOE shape information.
        _, unique_idx = np.unique(train_x, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return train_x[unique_idx], train_y[unique_idx]

    def _select_diverse_points(self, train_x, train_y, center):
        if len(train_x) <= self.inp_size:
            return train_x, train_y

        center = np.asarray(center).reshape(1, -1)
        dist_to_center = np.linalg.norm(train_x - center, axis=1)
        # Keep the DOE local to the current CMA state before enforcing diversity.
        candidate_count = min(len(train_x), max(self.inp_size * 4, self.inp_size + 1))
        candidate_idx = np.argsort(dist_to_center)[:candidate_count]
        candidate_x = train_x[candidate_idx]
        candidate_y = train_y[candidate_idx]
        candidate_center_dist = dist_to_center[candidate_idx]

        first_idx = int(np.argmin(candidate_center_dist))
        selected = [first_idx]
        min_pairwise_dist = np.linalg.norm(
            candidate_x - candidate_x[first_idx].reshape(1, -1), axis=1
        )

        while len(selected) < self.inp_size:
            min_pairwise_dist[selected] = -np.inf
            # Prefer points that expand x-space coverage, but bias slightly toward the CMA center.
            score = min_pairwise_dist - 0.05 * candidate_center_dist
            next_idx = int(np.argmax(score))
            if not np.isfinite(score[next_idx]):
                break
            selected.append(next_idx)
            dist_to_new = np.linalg.norm(
                candidate_x - candidate_x[next_idx].reshape(1, -1), axis=1
            )
            min_pairwise_dist = np.minimum(min_pairwise_dist, dist_to_new)

        selected = np.array(selected, dtype=int)
        selected = selected[np.argsort(candidate_center_dist[selected])]
        return candidate_x[selected], candidate_y[selected]

    def reset(self, dim):
        self.inp_size = int(dim*self.inp_size_base)
        # self.autoencoder.load_weights(f'{self.model_save_path}.h5')
        self.autoencoder = VAE(int(self.latent_dim*dim), self.inp_size, kl_weight=self.kl_weight)
        self.autoencoder.compile(optimizer="adam")
        self.active_functions = []
        self.distances = []
        self.old_xs = None


    def load_or_create(self, dim):
        self.reset(dim)
        if self.loaded: return self

        if (self.functions is None or len(self.functions) == 0):
            if os.path.exists(self.fun_save_path):
                self.functions = np.load(self.fun_save_path)[:self.n_functions]
            else:
                self.functions = self.generate_functions(self.gen_x_sample(10), self.functions)
                np.save(f"{self.fun_save_path}", self.functions)

        self.loaded = True
        return self

    def generate_functions(self, array_x, provided_functions=[]):
        def fun_gen():
            if provided_functions is not None:
                for f in provided_functions:
                    yield f
            while True:
                tree = genTree.generate_tree(6, 16)
                exp = genTree2exp.generate_tree2exp(tree)
                fun = genExp2fun.generate_exp2fun(exp)
                fun = '('+fun + ')[:,0]'
                yield fun

        functions = []
        orig_settings = np.seterr(all='raise')
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        iters = 0
        for fun in fun_gen():
            iters_per_succ = iters/max(len(functions),1)
            if len(functions) >= self.n_functions: break
            iters += 1
            try:
                array_y = eval(fun)
                if (
                    np.isnan(array_y).any()
                    or np.isinf(array_y).any()
                    or array_y.ndim != 1
                    or np.any(abs(array_y) < 1e-8)
                    or np.any(abs(array_y) > 1e8)
                    or len(np.unique(array_y)) < len(array_y)/1.5):
                        continue
                if (np.var(array_y) < 1.0):
                    if (np.var(array_y*10) < 1.0):
                        continue
                    else:
                        fun = '10*('+fun+')'
                functions.append(fun)
            except Exception as inst:
                continue
        warnings.simplefilter("default")
        np.seterr(**orig_settings)
        # assert(len(provided_functions) == 0 or iters==self.n)
        return np.array(functions)

    def gen_x_sample(self, dim):
        import math
        sampler = qmc.Sobol(d=dim, scramble=False, seed=self.seed)
        sample = sampler.random_base2(math.ceil(math.log2(self.inp_size)))
        sample = np.clip(sample, 0.001, 0.999)
        np.random.default_rng(self.seed).shuffle(sample)
        sample = sample[:self.inp_size]
        return sample



    def eval_functions(self, x):
        assert(np.sum(np.logical_or(x<0,x>1))==0)
        array_x = np.clip(x, 0.001, 0.999)
        functions = np.asarray(self.functions)
        if len(functions) == 0:
            self.active_functions = np.array([], dtype=object)
            return np.empty((0, len(array_x)))

        windows = list(group_list(functions, math.ceil(len(functions)/8)))


        y = self.executor.map(eval_multiple, windows, [array_x]*len(windows))
        y = [arr for arrs in list(y) for arr in arrs]  #flatten
        mask = np.array([a is not None for a in y], dtype=bool)
        y = np.array([a for a in y if a is not None])
        active_functions = functions[mask]

        valid_mask = np.sum(np.logical_or(np.isnan(y),np.isinf(y)), axis=-1)==0
        if y.size > 0:
            valid_mask = np.logical_and(valid_mask, np.ptp(y, axis=-1) > 1e-12)
        svm = np.sum(valid_mask)
        if svm/len(valid_mask) < 0.9:
            print()

        self.active_functions = active_functions[valid_mask]
        y = y[valid_mask,:]
        return y

    def fit(self, epochs=5,batch_size=128, val_n = 50, **kwargs):
        """Fit the autoencoder model.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            **kwargs (dict, optional): optional arguments for the fit procedure.
        """
        if self.autoencoder is None:
            raise AttributeError("Autoencoder model is not compiled yet")

        sample_count = int(self.Y.shape[0])
        if sample_count < 2:
            raise ValueError("Need at least two valid training functions for DOE autoencoder training")

        val_n = min(val_n, sample_count - 1)
        train_data = tf.cast(self.Y[:-val_n], tf.float32) if val_n > 0 else tf.cast(self.Y, tf.float32)
        validation_data = None
        if val_n > 0:
            validation = tf.cast(self.Y[-val_n:], tf.float32)
            validation_data = (validation, validation)

        # valid_mask = np.sum(np.logical_or(np.isnan(self.Y),np.isinf(self.Y)), axis=1)==0
        # self.Y = self.Y[valid_mask,:]
        # self.functions = self.functions[valid_mask]

        self.autoencoder.fit(
                train_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=validation_data,
                **kwargs
            )

    def train(self, train_x, train_y,opt=None):
        # self.approximation = lambda a: np.random.default_rng().random(a.shape[0])
        # return
        # start_time = timer()
        # mn = np.min(train_y)
        # mx = np.max(train_y)
        # train_y = (train_y - mn) / (mx-mn+(1e-4))
        # train_y = np.clip(train_y, 0.01, 0.99)

        # closest_xs = np.array(train_x)[-self.inp_size:]
        # closest_ys = np.array(train_y)[-self.inp_size:]

        train_x,train_y = np.array(train_x),np.array(train_y)
    # Build the DOE from unique points near the current CMA mean, then spread them in x-space.
        train_x, train_y = self._drop_duplicate_points(train_x, train_y)
        center = opt._mean if opt is not None and hasattr(opt, '_mean') else np.mean(train_x, axis=0)
        closest_xs, closest_ys = self._select_diverse_points(train_x, train_y, center)



        xs = np.clip((closest_xs+5)/10,0.01, 0.99)
        # mn = np.mean(closest_ys,axis=-1, keepdims=True)
        # std = np.std(closest_ys,axis=-1, keepdims=True)
        # closest_ys = (closest_ys - mn) / (np.where(std==0,1e-4,std))
        mn = np.min(closest_ys)
        mx = np.max(closest_ys)
        closest_ys = (closest_ys - mn) / (mx-mn+(1e-4))
        closest_ys = np.clip(closest_ys, 0.01, 0.99)

        #bipartite matching to minimize changes to inputs of the autoencoder
        if self.old_xs is not None:
            dist_matrix = distance_matrix(self.old_xs, xs)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            ordered_xs = xs[col_ind]
            closest_ys = closest_ys[col_ind]
            self.old_xs = ordered_xs
        else:
            self.old_xs = xs
        self.Y = self.eval_functions(self.old_xs)
        if len(self.active_functions) < 2:
            raise ValueError("Too few valid DOE functions remained for the current training round")

        # mn = np.min(y,axis=-1, keepdims=True)
        # mx = np.max(y,axis=-1, keepdims=True)
        # y = (y - mn) / ((mx - mn)+1e-4)
        # y = np.clip(y, 1e-3, 0.999)
        mn = np.min(self.Y,axis=-1, keepdims=True)
        mx = np.max(self.Y,axis=-1, keepdims=True)
        self.Y = (self.Y - mn) / (mx-mn+(1e-4))
        self.Y = np.clip(self.Y, 0.01, 0.99)


        # end_time = timer()
        # elapsed = end_time - start_time
        # print(f"evaluate funcs time: {elapsed}")

        self.fit(epochs=self.train_epochs)
        f,d = self.approximate(closest_ys, scale_inp=True)

        # end_time_ = timer()
        # elapsed = end_time_ - end_time
        # print(f"train time: {elapsed}")
        self.approximation = f
        self.distances.append(d)

    def __call__(self, xs):
        return self.approximation(xs)

    def approximate(self, array_y, scale_inp=True):
        # y evaluated from training funcs
        training_latent = self.encode(self.Y)
        # enc_min = np.min(training_latent,axis=0, keepdims=True)
        # enc_max = np.max(training_latent,axis=0, keepdims=True)
        # training_latent = (training_latent - enc_min) / ((enc_max - enc_min)+1e-4)
        # mn = np.mean(training_latent,axis=0, keepdims=True)
        # std = np.std(training_latent,axis=0, keepdims=True)
        # std = np.where(std==0, 1e-4, std)
        # training_latent = (training_latent - mn) / std # scale each column of the latent dim to make the nearestneighbor consider each node equally

        # y from the evo algorithm
        assert(len(array_y.shape)==1)
        latent = self.encode(array_y)
        if len(latent.shape)==1:
            latent = latent.reshape(1, -1)
        # latent = (latent - enc_min) / ((enc_max - enc_min)+1e-4)
        # latent = (latent - mn) / std


        # find closest function to use as an approximation
        eu_dists = np.linalg.norm(training_latent-latent, axis=1)
        i = np.argmin(eu_dists)
        mindist= eu_dists[i]
        print('approx distance', mindist)
        best_approx_str = self.active_functions[i]
        best_approx_f = eval('lambda array_x:'+best_approx_str)


        def run_approx(array_x):
            if (added_dim := len(array_x.shape)==1):
                array_x = array_x[np.newaxis,:]
            if scale_inp:
                array_x = (array_x + 5.0)/10 #scale from bbob vals to 0-1
                array_x = np.clip(array_x, 0.01, 0.99)
            e = best_approx_f(array_x)
            return e[0] if added_dim else e
        return run_approx, mindist

    def encode(self, y:np.ndarray):
        """Encode a Design of Experiments.

        Args:
            y (array): The DOE to encode.

        Returns:
            array: encoded feature vector.
        """

        if len(y.shape) == 1:
            y = y.reshape((1,-1))

        y_ = tf.cast(y, tf.float32)
        encoded_doe, _, __ = self.autoencoder.encoder(y_)
        encoded_doe = np.array(encoded_doe)
        encoded_doe = np.squeeze(encoded_doe)
        return encoded_doe


    def summary(self):
        """Get a summary of the autoencoder model"""
        self.autoencoder.encoder.summary()

    def plot_label_clusters_bbob(self):
        encodings = []
        fuction_groups = []
        for f in range(1, 25):
            for i in range(100):
                fun, opt = bbob.instantiate(f, i)
                bbob_y = np.asarray(list(map(fun, self.sample)))
                array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                    np.max(bbob_y) - np.min(bbob_y)
                )
                encoded = self.encode([array_x])
                encodings.append(encoded[0])
                fuction_groups.append(f)

        X = np.array(encodings)
        y = np.array(fuction_groups).flatten()
        mds = manifold.MDS(
            n_components=2,
            random_state=self.seed,
        )
        embedding = mds.fit_transform(X).T
        # display a 2D plot of the bbob functions in the latent space

        plt.figure(figsize=(12, 10))
        plt.scatter(embedding[0], embedding[1], c=y, cmap=cm.jet)
        plt.colorbar()
        plt.xlabel("")
        plt.ylabel("")

        if self.use_mlflow:
            plt.savefig("latent_space.png")
            mlflow.log_artifact("latent_space.png", "img")
        else:
            plt.savefig(
                f"latent_space_{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}.png"
            )

    def visualizeTestData(self, n=5):
        """Get a visualisation of the validation data.

        Args:
            n (int, optional): The number of validation DOEs to show. Defaults to 5.
        """
        if self.use_VAE:
            encoded_does, _z_log_var, _z = self.autoencoder.encoder(self.test_data)
        else:
            encoded_does = self.autoencoder.encoder(self.test_data).numpy()
        decoded_does = self.autoencoder.decoder(encoded_does).numpy()
        fig = plt.figure(figsize=(n * 4, 8))
        for i in range(n):
            # display original
            ax = fig.add_subplot(2, n, i + 1, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                self.test_data[i],
                cmap=cm.jet,
                antialiased=True,
            )
            no_descs(ax)
            plt.title("original")
            plt.gray()

            # display reconstruction
            ax = fig.add_subplot(2, n, i + 1 + n, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                decoded_does[i],
                cmap=cm.jet,
                antialiased=True,
            )
            no_descs(ax)
            plt.title("reconstructed")
            plt.gray()
        if self.use_mlflow:
            plt.savefig("reconstruction.png")
            mlflow.log_artifact("reconstruction.png", "img")
        else:
            plt.show()


if __name__ == "__main__":
    print()
    # import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # obj = doe_model(
    #     20,
    #     8,
    #     n=50000,
    #     latent_dim=40,
    #     kl_weight=0.001,
    #     use_mlflow=False,
    #     model_type="VAE",
    # )
    # obj.load_from_huggingface()
    # # test the model
    # obj.plot_label_clusters_bbob()
