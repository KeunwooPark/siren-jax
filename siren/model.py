from jax import jit
from jax import numpy as jnp
from siren.network import Siren
from siren.optimizer import minimize_with_jax_optim

class ImageModel:
    def __init__(self, layers):
        self.net = Siren(input_dim =2, layers=layers, output_dim = 3)
        self.loss_func = self.set_loss_func()

    def set_loss_func(self):

        @jit
        def loss_func(net_params, data):
            x = data['input']
            y = data['output']
            output = self.net.f(net_params, x)
            return jnp.mean((output - y)**2)

        return loss_func

    def update_net_params(self, params):
        self.net.net_params = params

    def get_params(self):
        return self.net.net_params
