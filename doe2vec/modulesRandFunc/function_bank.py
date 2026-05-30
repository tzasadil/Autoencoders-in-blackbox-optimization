from __future__ import annotations

import random
from functools import lru_cache

import jax
from jax import config as jax_config
import jax.numpy as jnp
import numpy as np


jax_config.update("jax_enable_x64", True)


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists

    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])

    return list_of_lists[:1] + flatten(list_of_lists[1:])


@lru_cache(maxsize=None)
def _rotation_matrix(seed, dim):
    return np.random.default_rng(seed).random((dim, dim))


@lru_cache(maxsize=None)
def _noise_column(seed, rows):
    return 1.0 + np.random.default_rng(seed).random((rows, 1)) / 10.0


def generate_function_spec(exp):
    exp_flat = flatten(exp)
    stack = []

    for item in exp_flat:
        if item < 0:
            stack.append(("const", float(abs(item))))
            continue

        if item == 1:
            stack.append(("const", float(random.random() * 9 + 1)))
        elif item == 2:
            stack.append(("input",))
        elif item == 3:
            stack.append(("first",))
        elif item == 4:
            stack.append(("shift",))
        elif item == 5:
            stack.append(("rotate", int(np.random.randint(0, 999999))))
        elif item == 6:
            stack.append(("index",))
        elif item == 7:
            stack.append(("noise", int(np.random.randint(0, 999999))))
        elif item in {11, 12, 13, 14}:
            right = stack.pop()
            left = stack.pop()
            stack.append(("binary", item, left, right))
        elif item in {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}:
            child = stack.pop()
            stack.append(("unary", item, child))
        else:
            raise ValueError(f"Operator {item} is not defined")

    if len(stack) != 1:
        raise ValueError("Invalid function specification")
    return stack[0]


def _evaluate_spec(spec, array_x):
    op = spec[0]

    if op == "const":
        return jnp.asarray(spec[1], dtype=array_x.dtype)
    if op == "input":
        return array_x
    if op == "first":
        return array_x[:, :1]
    if op == "shift":
        zeros = jnp.zeros((array_x.shape[0], 1), dtype=array_x.dtype)
        return jnp.concatenate((array_x[:, 1:], zeros), axis=1)
    if op == "rotate":
        matrix = jnp.asarray(_rotation_matrix(spec[1], int(array_x.shape[1])), dtype=array_x.dtype)
        return array_x @ matrix
    if op == "index":
        return jnp.arange(1, array_x.shape[1] + 1, dtype=array_x.dtype)
    if op == "noise":
        noise = jnp.asarray(_noise_column(spec[1], int(array_x.shape[0])), dtype=array_x.dtype)
        return noise

    if op == "binary":
        code = spec[1]
        left = _evaluate_spec(spec[2], array_x)
        right = _evaluate_spec(spec[3], array_x)
        if code == 11:
            return left + right
        if code == 12:
            return left - right
        if code == 13:
            return left * right
        if code == 14:
            return left / right

    if op == "unary":
        code = spec[1]
        child = _evaluate_spec(spec[2], array_x)
        if code == 21:
            return -child
        if code == 22:
            return 1 / child
        if code == 23:
            return 10 * child
        if code == 24:
            return jnp.square(child)
        if code == 25:
            return jnp.sqrt(jnp.abs(child))
        if code == 26:
            return jnp.abs(child)
        if code == 27:
            return jnp.round(child)
        if code == 28:
            return jnp.sin(2 * jnp.pi * child)
        if code == 29:
            return jnp.cos(2 * jnp.pi * child)
        if code == 30:
            return jnp.log(jnp.abs(child))
        if code == 31:
            return jnp.exp(child)
        if code == 32:
            return jnp.sum(child, axis=1)[:, jnp.newaxis]
        if code == 33:
            return jnp.mean(child, axis=1)[:, jnp.newaxis]
        if code == 34:
            return jnp.cumsum(child, axis=-1)
        if code == 35:
            return jnp.prod(child, axis=1)[:, jnp.newaxis]
        if code == 36:
            return jnp.amax(child, axis=1)[:, jnp.newaxis]

    raise ValueError(f"Unsupported function op: {op}")


def compile_function_spec(spec):
    def run(array_x):
        array_x = jnp.asarray(array_x, dtype=jnp.float64)
        result = _evaluate_spec(spec, array_x)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        return result

    return run


def precompile_function_spec(spec, input_shape):
    compiled = jax.jit(compile_function_spec(spec))
    sample_input = jnp.zeros(tuple(input_shape), dtype=jnp.float64)
    compiled(sample_input).block_until_ready()
    return compiled