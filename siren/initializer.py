from jax import random
from jax._src.nn.initializers import _compute_fans
from jax import numpy as jnp

def siren_init(omega = 30, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(6 / fan_in) / omega
        return random.uniform(key, shape, dtype, minval = -variance, maxval = variance)

    return init

def siren_init_first(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = 1 / fan_in
    return random.uniform(key, shape, dtype, minval = -variance, maxval = variance)

    return init

def bias_uniform(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    # this is what Pytorch default Linear uses.
    def init(key, shape, dtype=dtype):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(1 / fan_in)
        return random.uniform(key, (int(fan_out),), dtype, minval = -variance, maxval = variance)
    return init
