from jax import jit
from jax import numpy as jnp
from siren.network import Siren
from siren.optimizer import minimize_with_jax_optim

class ImageModel:
    def __init__(self, layers):
        self.net = Siren(input_dim =2, layers, output_dim = 3)

    def get_loss_func(self, data):
        x = data['input']
        y = data['output']

        @jit
        def loss_func(net_params):
            output = self.net.get_f_p(x)(net_params)
            return jnp.mean((output - y)**2)

        return loss_func

    def update_net_params(self, params):
        self.net.net_params = params
