from jax.experimental import stax
import jax
from jax import vmap, jacfwd, hessian
from jax import numpy as jnp

from siren.initializer import siren_init, siren_init_first, bias_uniform
from siren import layer
import time

def create_mlp(input_dim, num_channels, output_dim, omega = 30):
    modules = []
    modules.append(layer.Dense(num_channels[0], W_init = siren_init_first(), b_init=bias_uniform()))
    modules.append(layer.Sine(omega))
    for nc in num_channels[1:]:
        modules.append(layer.Dense(nc, W_init = siren_init(omega = omega), b_init=bias_uniform()))
        modules.append(layer.Sine(omega))

    modules.append(layer.Dense(output_dim, W_init = siren_init(omega = omega), b_init=bias_uniform()))
    net_init_random, net_apply = stax.serial(*modules)

    in_shape = (-1, input_dim)
    rng = create_random_generator()
    out_shape, net_params = net_init_random(rng, in_shape)

    return net_params, net_apply

def create_random_generator(rng_seed=None):
    if rng_seed is None:
        rng_seed = int(round(time.time()))
    rng = jax.random.PRNGKey(rng_seed)

    return rng

class Siren:
    def __init__(
        self, input_dim, layers, output_dim, omega, rng_seed=None
    ):
        net_params, net_apply = create_mlp(input_dim, layers, output_dim, omega)

        self.net_params = net_params
        self.net_apply = net_apply

    """ *_p methods are used for optimization """
    def f(self, net_params, x):
        return self.net_apply(net_params, x)

    def df(self, net_params, x):
        return elementwise_jacobian(self.net_apply, net_params, x)

    def d2f(self, net_params, x):
        return hessian_wrt_input(self.net_apply, self.net_params, x)

def elementwise_jacobian(net_apply, net_params, x):
    f = lambda x: net_apply(net_params, x)
    y, f_vjp = jax.vjp(f, x)
    x_grad, = f_vjp(jnp.ones_like(y))
    x_grad = jnp.expand_dims(x_grad, axis = 1)
    return x_grad

def jacobian_wrt_input(net_apply, net_params, x):
    f = lambda x: net_apply(net_params, x)
    vmap_jac = vmap(jacfwd(f))

    J = vmap_jac(x)
    return J

def hessian_wrt_input(net_apply, net_params, x):
    f = lambda x: net_apply(net_params, x)
    vmap_hessain = vmap(hessian(f))
    H = vmap_hessain(x)
    h_diag = H.diagonal(0, 2, 3)

    return h_diag
