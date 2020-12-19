from jax import numpy as jnp
from jax.nn.initializers import glorot_normal
from jax.experimental import stax

def Dense(out_dim, W_init=glorot_normal(), b_init=glorot_normal()):
    """(Custom) Layer constructor function for a dense (fully-connected) layer."""
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        # the below line is different from the original jax's Dense
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (input_shape[-1], out_dim))
        return output_shape, (W, b)
    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W) + b
    return init_fun, apply_fun

def sine_with_gain(gain):
    def func(input):
        return jnp.sin(input * gain)

def Sine(gain):
    return stax.elementwize(sine_with_gain(gain))
