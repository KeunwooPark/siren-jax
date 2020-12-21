from abc import ABC, abstractmethod
import jax
from jax.experimental import stax, optimizers
from jax import jit
from tensorflow_probability.substrates import jax as tfp
from jax import numpy as jnp
import numpy as np
import time

class TrainingState:
    def __init__(self, params):
        self.params = params
        self.iter = 0
        self.layers_shape = None
        self.layers_grad_mean = None
        self.duration_per_iter = 0

jax_optimizers = ["adam"]
scpiy_style_optimizers = ["l-bfgs"]

def minimize_with_jax_optim(
    name, loss_func, start_point, lr, iter, interm_iter, training_state, callback
):
    if name not in jax_optimizers:
        raise ValueError("Optimizer {} is not implemented yet".format(name))

    if name == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    params = start_point
    opt_state = opt_init(params)

    @jit
    def step(i, _opt_state):
        _params = get_params(_opt_state)
        g = jax.grad(loss_func)(_params)
        return opt_update(i, g, _opt_state)

    timestamp = time.perf_counter()
    for i in range(iter):
        opt_state = step(i, opt_state)
        training_state.iter = i + 1
        if (i + 1) % interm_iter == 0:
            param = get_params(opt_state)
            training_state.params = param
            duration_per_iter = (time.perf_counter() - timestamp) / interm_iter
            training_state.duration_per_iter = duration_per_iter
            timestamp = time.perf_counter()
            callback(training_state)

    param = get_params(opt_state)
    training_state.params = param
    return param

def get_layers_mean(grad):
    g_flatten, g_tree = jax.tree_util.tree_flatten(grad)
    means = []
    shapes = []
    for g in g_flatten:
        means.append(float(jnp.mean(jnp.abs(g))))
        shapes.append(g.shape)

    return shapes, means

def get_tfp_minimize_func(name):
    if name == "l-bfgs":
        return tfp.optimizer.lbfgs_minimize

    raise NotImplementedError("{} is not implemented".format(name))


def concat_net_params(params):
    # reference: https://github.com/google/jax/issues/1400#issuecomment-710378214
    flat_params, params_tree = jax.tree_util.tree_flatten(params)
    params_shape = [x.shape for x in flat_params]
    return jnp.concatenate([x.reshape(-1) for x in flat_params]), (
        params_tree,
        params_shape,
    )


def split_net_params(param_vector, params_tree, params_shape):
    split_point_idx = np.cumsum([np.prod(s) for s in params_shape[:-1]])
    splitted_params = jnp.split(param_vector, split_point_idx)

    splitted_flat_params = [x.reshape(s) for x, s in zip(splitted_params, params_shape)]
    params = jax.tree_util.tree_unflatten(params_tree, splitted_flat_params)
    return params


def minimize_with_tfp_optim(
    name,
    loss_func,
    start_point,
    max_iter,
    interm_iter,
    training_state,
    callback,
    tol=1e-8,
):
    if max_iter == 0:
        return start_point

    num_iter_chunk = int(max_iter / interm_iter)
    residual_iter = max_iter - num_iter_chunk * interm_iter

    opt = get_tfp_minimize_func(name)
    param_vector, (params_tree, params_shape) = concat_net_params(start_point)

    @jax.value_and_grad
    def wrapper_func(param_vector):
        params = split_net_params(param_vector, params_tree, params_shape)
        return loss_func(params)

    for i in range(num_iter_chunk):
        result = opt(
            jax.jit(wrapper_func),
            initial_position=param_vector,
            tolerance=tol,
            max_iterations=interm_iter,
        )
        param_vector = result.position
        params = split_net_params(param_vector, params_tree, params_shape)

        training_state.iter += result.num_iterations
        training_state.params = params
        training_state.converged = result.converged
        training_state.failed = result.failed
        callback(training_state)

        if result.num_iterations < interm_iter:  # early finish
            return params
        if result.converged or result.failed:
            return params

    if residual_iter > 0:
        result = opt(
            jax.jit(wrapper_func),
            initial_position=param_vector,
            tolerance=tol,
            max_iterations=residual_iter,
        )
        param_vector = result.position
        params = split_net_params(param_vector, params_tree, params_shape)

        training_state.iter += result.num_iterations
        training_state.params = params
        callback(training_state)

    return params
