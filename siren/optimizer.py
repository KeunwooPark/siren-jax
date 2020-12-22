from abc import ABC, abstractmethod
import jax
from jax.experimental import stax, optimizers
from jax import jit
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

def minimize_with_jax_optim(name, model, data_loader, training_state, lr, epoch, print_iter, callback):
    if name not in jax_optimizers:
        raise ValueError("Optimizer {} is not implemented yet".format(name))

    if name == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    params = training_state.params
    opt_state = opt_init(params)

    @jit
    def step(i, _opt_state, data):
        _params = get_params(_opt_state)
        loss_func = model.get_loss_func(data)
        g = jax.grad(loss_func)(_params)
        return opt_update(i, g, _opt_state)

    timestamp = time.perf_counter()
    iter = 0
    for _ in range(epoch):
        for data in data_loader:
            opt_state = step(iter, opt_state, data)
            iter += 1
            training_state.iter = iter
            if (iter) % print_iter == 0:
                param = get_params(opt_state)
                training_state.params = param
                duration_per_iter = (time.perf_counter() - timestamp) / interm_iter
                training_state.duration_per_iter = duration_per_iter
                timestamp = time.perf_counter()
                callback(training_state)

        data_loader.shuffle()

    # when # of iter cannot be divided into print_iter
    if not iter % print_iter == 0:
        training_state.iter = iter
        param = get_params(opt_state)
        training_state.params = param
        residual_iter = iter % print_iter
        duration_per_iter = (time.perf_counter() - timestamp) / residual_iter
        callback(training_state)

    return training_state

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
