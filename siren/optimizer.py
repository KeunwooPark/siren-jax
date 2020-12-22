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

class JaxOptimizer:
    def __init__(self, optimizer_name, model, lr):
        if optimizer_name not in jax_optimizers:
            raise ValueError("Optimizer {} is not implemented yet".format(name))

        if optimizer_name == "adam":
            opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

        self.opt_state = opt_init(model.get_params())
        self.get_params = get_params

        @jit
        def step(i, _opt_state, data):
            _params = get_params(_opt_state)
            loss_func = model.loss_func
            g = jit(jax.grad(loss_func))(_params, data)
            return opt_update(i, g, _opt_state)

        self._step = step
        self.iter_cnt = 0

    def step(self, data):
        self.opt_state = self._step(self.iter_cnt, self.opt_state, data)
        self.iter_cnt += 1

    def get_optimized_params(self):
        return self.get_params(self.opt_state)


def minimize_with_jax_optim(name, model, data_loader, lr, epoch, print_iter, callback):
    if name not in jax_optimizers:
        raise ValueError("Optimizer {} is not implemented yet".format(name))

    if name == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    training_state = TrainingState(model.get_params())

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
    last_data = None
    for _ in range(epoch):
        data_loader.create_batches()
        for data in data_loader:
            opt_state = step(iter, opt_state, data)
            iter += 1
            training_state.iter = iter
            last_data = data
            print(_, iter)
            if (iter) % print_iter == 0:
                param = get_params(opt_state)
                training_state.params = param
                duration_per_iter = (time.perf_counter() - timestamp) / interm_iter
                training_state.duration_per_iter = duration_per_iter
                timestamp = time.perf_counter()
                callback(data, training_state)

    # when # of iter cannot be divided into print_iter
    if not iter % print_iter == 0:
        training_state.iter = iter
        param = get_params(opt_state)
        training_state.params = param
        residual_iter = iter % print_iter
        duration_per_iter = (time.perf_counter() - timestamp) / residual_iter
        callback(last_data, training_state)

    return training_state
