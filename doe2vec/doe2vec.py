import os
import sys
import warnings
from dataclasses import dataclass
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
from time import perf_counter
from scipy.stats import qmc
from sklearn import manifold
import math
# from doe2vec import bbobbenchmarks as bbob
from doe2vec.vae import VAE
from doe2vec.modulesRandFunc.function_bank import (
    compile_function_spec,
    generate_function_spec,
    precompile_function_spec,
)
from doe2vec.modulesRandFunc import generate_tree as genTree
from doe2vec.modulesRandFunc import generate_tree2exp as genTree2exp
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment, minimize

def no_descs(ax):
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            for line in axis.get_ticklines():
                line.set_visible(False)


def _zscore_row_static(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    std = np.std(values)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


def _rotation_pairs_static(dim):
    return [(left, right) for left in range(dim - 1) for right in range(left + 1, dim)]


def _rotation_matrix_static(dim, angles, angle_pairs):
    rotation = np.eye(dim)
    for angle, (left, right) in zip(angles, angle_pairs):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        givens = np.eye(dim)
        givens[left, left] = cosine
        givens[left, right] = -sine
        givens[right, left] = sine
        givens[right, right] = cosine
        rotation = givens @ rotation
    return rotation


def _apply_transform_static(unit_xs, center_unit, translation, angles, angle_pairs):
    centered = unit_xs - center_unit.reshape(1, -1)
    rotation = _rotation_matrix_static(unit_xs.shape[1], angles, angle_pairs)
    transformed = center_unit.reshape(1, -1) + translation.reshape(1, -1) + centered @ rotation.T
    return np.clip(transformed, 0.01, 0.99)


def _fit_function_transform_worker(task):
    (
        function_idx,
        function,
        compiled_function,
        unit_xs,
        target_values,
        center_unit,
        translation_bound,
        transform_maxfev,
        cached_params,
    ) = task
    if compiled_function is None:
        return None

    dim = unit_xs.shape[1]
    angle_pairs = _rotation_pairs_static(dim)
    target_z = _zscore_row_static(target_values)
    bounds = [(-translation_bound, translation_bound)] * dim + [(-math.pi, math.pi)] * len(angle_pairs)
    zero_params = np.zeros(dim + len(angle_pairs), dtype=float)
    initial_params = (
        np.asarray(cached_params, dtype=float).copy()
        if cached_params is not None and len(cached_params) == len(zero_params)
        else zero_params.copy()
    )

    def evaluate_params(params):
        translation = np.asarray(params[:dim], dtype=float)
        angles = np.asarray(params[dim:], dtype=float)
        transformed_xs = _apply_transform_static(
            unit_xs, center_unit, translation, angles, angle_pairs
        )
        try:
            outputs = np.asarray(compiled_function(transformed_xs), dtype=float).reshape(-1)
        except Exception:
            return None, np.inf
        if outputs.shape[0] != unit_xs.shape[0]:
            return None, np.inf
        if np.any(~np.isfinite(outputs)) or np.ptp(outputs) <= 1e-12:
            return None, np.inf
        loss = np.mean((_zscore_row_static(outputs) - target_z) ** 2)
        if not np.isfinite(loss):
            return None, np.inf
        return outputs, float(loss)

    best_outputs, best_loss = evaluate_params(initial_params)
    best_params = initial_params.copy()
    if best_outputs is None and np.any(initial_params != 0.0):
        best_outputs, best_loss = evaluate_params(zero_params)
        best_params = zero_params.copy()
    if best_outputs is None:
        return None

    if transform_maxfev > 1 and best_params.size > 0:
        try:
            result = minimize(
                lambda params: evaluate_params(params)[1],
                best_params,
                method="Powell",
                bounds=bounds,
                options={"maxfev": transform_maxfev, "disp": False},
            )
            candidate_params = np.asarray(result.x, dtype=float) if result.x is not None else best_params
            candidate_outputs, candidate_loss = evaluate_params(candidate_params)
            if candidate_outputs is not None and candidate_loss < best_loss:
                best_outputs = candidate_outputs
                best_loss = candidate_loss
                best_params = candidate_params
        except Exception:
            pass

    return FittedFunctionModel(
        function=function,
        callable=compiled_function,
        center=center_unit.copy(),
        function_idx=int(function_idx),
        translation=best_params[:dim].copy(),
        angles=best_params[dim:].copy(),
        angle_pairs=angle_pairs,
        loss=best_loss,
        outputs=best_outputs,
    )


@dataclass
class FittedFunctionModel:
    function: object
    callable: object
    center: np.ndarray
    function_idx: int | None
    translation: np.ndarray
    angles: np.ndarray
    angle_pairs: list[tuple[int, int]]
    loss: float = math.inf
    outputs: np.ndarray | None = None

class doe_model:
    def __init__(
        self,
        inp_size,
        latent_dim,
        n_functions=1_000,
        seed_nr=0,
        kl_weight=0.001,
        preserve_input_order=True,
        drop_duplicate_points=True,
        point_selection="local_diverse",
        transform_maxfev=40,
        translation_bound=0.2,
        use_transform_fitting=True,
        selector_mode="latent",
        precompile_bank_functions=True,
        autoencoder_batch_divisor=20,
    ):
        """Encode local DOEs with a VAE and pick a nearby bank function as surrogate.

        point_selection: "local_diverse" or "local_nearest".
        selector_mode: "latent" or "fitted_loss".
        """
        self.inp_size_base = inp_size
        self.n_functions = n_functions
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.seed = seed_nr
        self.loaded = False
        self.autoencoder = None
        self.Y = np.empty((0, 0), dtype=float)
        self.functions = []
        self.compiled_functions = []
        self.active_functions = []
        self.active_compiled_functions = []
        self.active_function_models = []
        self.distances = []
        self.transform_cache = {}
        self.bank_iteration = 0
        self.bank_ranked_function_indices = np.array([], dtype=int)
        self.last_bank_fit_was_full = True
        self.last_bank_fit_candidate_count = 0
        self.fun_save_path = f'doe_saves/functions_jax.npy'
        # self.model_save_path = f'doe_saves/{self.inp_size}_{self.latent_dim}'
        seed(self.seed)
        # worker_n = 8
        # self.worker_conns, child_conns = list(zip(*[Pipe() for _ in range(worker_n)]))
        # self.eval_workers = [Process(target=evaluator, args=(conn,)) for conn in child_conns]
        # for p in self.eval_workers: p.start()


        self.train_epochs = 1
        self.old_xs = None
        self.preserve_input_order = bool(preserve_input_order)
        self.drop_duplicate_points = bool(drop_duplicate_points)
        self.point_selection = point_selection
        self.transform_maxfev = int(transform_maxfev)
        self.translation_bound = float(translation_bound)
        self.use_transform_fitting = bool(use_transform_fitting)
        self.selector_mode = str(selector_mode)
        self.precompile_bank_functions = bool(precompile_bank_functions)
        self.autoencoder_batch_divisor = max(1, int(autoencoder_batch_divisor))
        if self.point_selection not in {"local_diverse", "local_nearest"}:
            raise ValueError(f"Unsupported point_selection: {self.point_selection}")
        if self.selector_mode not in {"latent", "fitted_loss"}:
            raise ValueError(f"Unsupported selector_mode: {self.selector_mode}")


    def __str__(self):
        return f'doe_{self.inp_size_base}_{self.latent_dim}'

    def _drop_duplicate_points(self, train_x, train_y):
        if len(train_x) == 0:
            return train_x, train_y
        # drop duplicate x's; they don't change the DOE shape
        _, unique_idx = np.unique(train_x, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return train_x[unique_idx], train_y[unique_idx]

    def _select_diverse_points(self, train_x, train_y, center):
        if len(train_x) <= self.inp_size:
            return train_x, train_y

        center = np.asarray(center).reshape(1, -1)
        dist_to_center = np.linalg.norm(train_x - center, axis=1)
        # first restrict to a neighborhood of the CMA mean
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
            # max-min distance, with a small pull toward the center
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

    def _select_local_points(self, train_x, train_y, center):
        if len(train_x) <= self.inp_size:
            return train_x, train_y

        center = np.asarray(center).reshape(1, -1)
        dist_to_center = np.linalg.norm(train_x - center, axis=1)
        if self.point_selection == "local_nearest":
            nearest_idx = np.argsort(dist_to_center)[: self.inp_size]
            nearest_idx = nearest_idx[np.argsort(dist_to_center[nearest_idx])]
            return train_x[nearest_idx], train_y[nearest_idx]
        return self._select_diverse_points(train_x, train_y, center)

    def reset(self, dim):
        self.inp_size = int(dim*self.inp_size_base)
        # self.autoencoder.load_weights(f'{self.model_save_path}.h5')
        self.autoencoder = VAE(int(self.latent_dim*dim), self.inp_size, kl_weight=self.kl_weight)
        self.autoencoder.compile(optimizer="adam")
        self.Y = np.empty((0, self.inp_size), dtype=float)
        self.active_functions = []
        self.active_compiled_functions = []
        self.active_function_models = []
        self.distances = []
        self.transform_cache = {}
        self.bank_iteration = 0
        self.bank_ranked_function_indices = np.array([], dtype=int)
        self.last_bank_fit_was_full = True
        self.last_bank_fit_candidate_count = 0
        self.old_xs = None


    def load_or_create(self, dim):
        self.reset(dim)
        if self.loaded: return self

        if (self.functions is None or len(self.functions) == 0):
            if os.path.exists(self.fun_save_path):
                self.functions = np.load(self.fun_save_path, allow_pickle=True)[:self.n_functions]
            else:
                self.functions = self.generate_functions(self.gen_x_sample(10), self.functions)
                np.save(f"{self.fun_save_path}", np.asarray(self.functions, dtype=object))
        self.compiled_functions = self._compile_functions(
            self.functions,
            input_shape=(self.inp_size, dim),
        )

        self.loaded = True
        return self

    def _compile_functions(self, functions, input_shape=None):
        compiled = []
        for function in np.asarray(functions):
            try:
                if self.precompile_bank_functions and input_shape is not None:
                    compiled.append(precompile_function_spec(function, input_shape))
                else:
                    compiled.append(compile_function_spec(function))
            except Exception:
                compiled.append(None)
        return np.array(compiled, dtype=object)

    def _normalize_input_points(self, xs):
        xs = np.asarray(xs, dtype=float)
        return np.clip((xs + 5.0) / 10.0, 0.01, 0.99)

    def _normalize_value_row(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        mn = np.min(values)
        mx = np.max(values)
        normalized = (values - mn) / (mx - mn + 1e-4)
        return np.clip(normalized, 0.01, 0.99)

    def _zscore_row(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        std = np.std(values)
        if not np.isfinite(std) or std <= 1e-12:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std

    def _rotation_pairs(self, dim):
        return [(left, right) for left in range(dim - 1) for right in range(left + 1, dim)]

    def _rotation_matrix(self, dim, angles, angle_pairs):
        rotation = np.eye(dim)
        for angle, (left, right) in zip(angles, angle_pairs):
            cosine = np.cos(angle)
            sine = np.sin(angle)
            givens = np.eye(dim)
            givens[left, left] = cosine
            givens[left, right] = -sine
            givens[right, left] = sine
            givens[right, right] = cosine
            rotation = givens @ rotation
        return rotation

    def _apply_transform(self, unit_xs, center_unit, translation, angles, angle_pairs):
        centered = unit_xs - center_unit.reshape(1, -1)
        rotation = self._rotation_matrix(unit_xs.shape[1], angles, angle_pairs)
        transformed = center_unit.reshape(1, -1) + translation.reshape(1, -1) + centered @ rotation.T
        return np.clip(transformed, 0.01, 0.99)

    def _select_bank_candidate_indices(self):
        total_functions = len(self.functions)
        if total_functions == 0:
            return np.array([], dtype=int)
        return np.arange(total_functions, dtype=int)

    def _fit_function_bank(self, unit_xs, target_values, center, candidate_indices=None):
        if not self.use_transform_fitting:
            raw_rows = self.eval_functions(unit_xs)
            center_unit = self._normalize_input_points(np.asarray(center).reshape(1, -1))[0]
            angle_pairs = self._rotation_pairs(unit_xs.shape[1])
            self.active_function_models = [
                FittedFunctionModel(
                    function=function,
                    callable=compiled_function,
                    center=center_unit.copy(),
                    function_idx=function_idx,
                    translation=np.zeros(unit_xs.shape[1], dtype=float),
                    angles=np.zeros(len(angle_pairs), dtype=float),
                    angle_pairs=angle_pairs,
                )
                for function_idx, (function, compiled_function) in enumerate(
                    zip(self.active_functions, self.active_compiled_functions)
                )
            ]
            if raw_rows.size == 0:
                return raw_rows
            return np.asarray([self._normalize_value_row(row) for row in raw_rows], dtype=float)

        center_unit = self._normalize_input_points(np.asarray(center).reshape(1, -1))[0]
        if candidate_indices is None:
            candidate_indices = np.arange(len(self.functions), dtype=int)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int)
        fit_tasks = [
            (
                function_idx,
                self.functions[function_idx],
                self.compiled_functions[function_idx],
                unit_xs,
                target_values,
                center_unit,
                self.translation_bound,
                self.transform_maxfev,
                None
                if self.transform_cache.get(function_idx) is None
                else np.asarray(self.transform_cache[function_idx], dtype=float).copy(),
            )
            for function_idx in candidate_indices
        ]
        fitted_candidates = map(_fit_function_transform_worker, fit_tasks)

        fitted_rows = []
        active_functions = []
        active_compiled_functions = []
        active_models = []
        for fitted_model in fitted_candidates:
            if fitted_model is None:
                continue
            self.transform_cache[fitted_model.function_idx] = np.concatenate(
                [fitted_model.translation, fitted_model.angles]
            )
            fitted_rows.append(self._normalize_value_row(fitted_model.outputs))
            active_functions.append(fitted_model.function)
            active_compiled_functions.append(fitted_model.callable)
            active_models.append(fitted_model)

        self.active_functions = np.array(active_functions, dtype=object)
        self.active_compiled_functions = np.array(active_compiled_functions, dtype=object)
        self.active_function_models = active_models
        if len(fitted_rows) == 0:
            return np.empty((0, len(unit_xs)))
        return np.asarray(fitted_rows, dtype=float)

    def generate_functions(self, array_x, provided_functions=None):
        def fun_gen():
            if provided_functions is not None:
                for function_spec in provided_functions:
                    yield function_spec
            while True:
                tree = genTree.generate_tree(6, 16)
                exp = genTree2exp.generate_tree2exp(tree)
                yield generate_function_spec(exp)

        functions = []
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        iters = 0
        for function_spec in fun_gen():
            iters_per_succ = iters/max(len(functions),1)
            if len(functions) >= self.n_functions: break
            iters += 1
            try:
                compiled_function = compile_function_spec(function_spec)
                array_y = np.asarray(compiled_function(array_x), dtype=float)
                if (
                    np.isnan(array_y).any()
                    or np.isinf(array_y).any()
                    or array_y.ndim != 1
                    or np.any(abs(array_y) < 1e-8)
                    or np.any(abs(array_y) > 1e8)
                    or len(np.unique(array_y)) < len(array_y)/1.5):
                        continue
                if (np.var(array_y) < 1.0):
                    scaled_spec = ("unary", 23, function_spec)
                    scaled_y = np.asarray(compile_function_spec(scaled_spec)(array_x), dtype=float)
                    if (np.var(scaled_y) < 1.0):
                        continue
                    function_spec = scaled_spec
                functions.append(function_spec)
            except Exception as inst:
                continue
        warnings.simplefilter("default")
        return np.array(functions, dtype=object)

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
            self.active_compiled_functions = np.array([], dtype=object)
            return np.empty((0, len(array_x)))

        rows = []
        active_functions = []
        active_compiled_functions = []
        for function, compiled_function in zip(functions, self.compiled_functions):
            if compiled_function is None:
                continue
            try:
                values = np.asarray(compiled_function(array_x), dtype=float)
            except Exception:
                continue
            if values.ndim != 1 or values.shape[0] != len(array_x):
                continue
            if np.any(~np.isfinite(values)) or np.ptp(values) <= 1e-12:
                continue
            rows.append(values)
            active_functions.append(function)
            active_compiled_functions.append(compiled_function)

        self.active_functions = np.array(active_functions, dtype=object)
        self.active_compiled_functions = np.array(active_compiled_functions, dtype=object)
        if len(rows) == 0:
            return np.empty((0, len(array_x)))
        return np.asarray(rows, dtype=float)

    def fit(self, epochs=5, batch_size=None, val_n=50, **kwargs):
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
        train_count = int(train_data.shape[0])
        resolved_batch_size = batch_size
        if resolved_batch_size is None:
            resolved_batch_size = max(
                1,
                int(math.ceil(max(train_count, 1) / self.autoencoder_batch_divisor)),
            )
        resolved_batch_size = min(train_count, int(resolved_batch_size)) if train_count > 0 else 1
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
            batch_size=resolved_batch_size,
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

        iteration_start = perf_counter()
        train_x,train_y = np.array(train_x),np.array(train_y)
        if self.drop_duplicate_points:
            train_x, train_y = self._drop_duplicate_points(train_x, train_y)
        center = opt._mean if opt is not None and hasattr(opt, '_mean') else np.mean(train_x, axis=0)
        closest_xs, closest_ys = self._select_local_points(train_x, train_y, center)



        xs = self._normalize_input_points(closest_xs)
        target_values = np.asarray(closest_ys, dtype=float).reshape(-1)
        normalized_target_values = self._normalize_value_row(target_values)

        # match previous DOE order when preserve_input_order is on
        if self.preserve_input_order and self.old_xs is not None:
            dist_matrix = distance_matrix(self.old_xs, xs)
            _row_ind, col_ind = linear_sum_assignment(dist_matrix)
            ordered_xs = xs[col_ind]
            target_values = target_values[col_ind]
            normalized_target_values = normalized_target_values[col_ind]
            self.old_xs = ordered_xs
            eval_xs = self.old_xs
        elif self.preserve_input_order:
            self.old_xs = xs
            eval_xs = self.old_xs
        else:
            self.old_xs = None
            eval_xs = xs
        fit_bank_start = perf_counter()
        candidate_indices = self._select_bank_candidate_indices()
        self.last_bank_fit_was_full = True
        self.last_bank_fit_candidate_count = len(candidate_indices)
        self.Y = self._fit_function_bank(
            eval_xs,
            target_values,
            center,
            candidate_indices=candidate_indices,
        )
        fit_bank_elapsed = perf_counter() - fit_bank_start
        print(
            f'fit function bank time {fit_bank_elapsed:.3f}s '
            f'(full, candidates={self.last_bank_fit_candidate_count})'
        )
        if len(self.active_functions) < 2:
            raise ValueError("Too few valid DOE functions remained for the current training round")


        # end_time = timer()
        # elapsed = end_time - start_time
        # print(f"evaluate funcs time: {elapsed}")

        autoencoder_fit_start = perf_counter()
        self.fit(epochs=self.train_epochs)
        autoencoder_fit_elapsed = perf_counter() - autoencoder_fit_start
        approx_start = perf_counter()
        f,d = self.approximate(normalized_target_values, scale_inp=True)
        approx_elapsed = perf_counter() - approx_start
        iteration_elapsed = perf_counter() - iteration_start
        post_bank_elapsed = iteration_elapsed - fit_bank_elapsed
        overlapped_iteration_floor = max(fit_bank_elapsed, post_bank_elapsed)
        async_savings_ceiling = iteration_elapsed - overlapped_iteration_floor
        print(
            "doe iteration time "
            f"total={iteration_elapsed:.3f}s "
            f"bank={fit_bank_elapsed:.3f}s "
            f"ae_fit={autoencoder_fit_elapsed:.3f}s "
            f"approx={approx_elapsed:.3f}s "
            f"post_bank={post_bank_elapsed:.3f}s "
            f"async_floor={overlapped_iteration_floor:.3f}s "
            f"async_max_savings={async_savings_ceiling:.3f}s "
            f"bank_rows={len(self.Y)}"
        )

        # end_time_ = timer()
        # elapsed = end_time_ - end_time
        # print(f"train time: {elapsed}")
        self.approximation = f
        self.distances.append(d)
        self.bank_iteration += 1

    def __call__(self, xs):
        return self.approximation(xs)

    def approximate(self, array_y, scale_inp=True):
        if self.selector_mode == "fitted_loss":
            losses = np.array([model.loss for model in self.active_function_models], dtype=float)
            ranking = np.argsort(losses)
            i = int(np.argmin(losses))
            mindist = float(losses[i])
            print('approx fitted loss', mindist)
        else:
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
            ranking = np.argsort(eu_dists)
            i = np.argmin(eu_dists)
            mindist= eu_dists[i]
            print('approx distance', mindist)
        self.bank_ranked_function_indices = np.asarray(
            [self.active_function_models[idx].function_idx for idx in ranking],
            dtype=int,
        )
        best_approx = self.active_function_models[i]


        def run_approx(array_x):
            if (added_dim := len(array_x.shape)==1):
                array_x = array_x[np.newaxis,:]
            if scale_inp:
                array_x = self._normalize_input_points(array_x)
            transformed_xs = self._apply_transform(
                array_x,
                best_approx.center,
                best_approx.translation,
                best_approx.angles,
                best_approx.angle_pairs,
            )
            e = best_approx.callable(transformed_xs)
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
