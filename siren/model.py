from jax import jit
from jax import numpy as jnp
from siren.network import Siren
from abc import ABC, abstractmethod

def get_model_cls_by_type(type):
    if type == 'normal':
        return NormalImageModel
    elif type == 'gradient':
        return GradientImageModel
    elif type == 'laplacian':
        return LaplacianImageModel
    
    raise ValueError("Wrong model type {}".format(type))

class BaseImageModel(ABC):
    def __init__(self, layers, n_channel, omega):
        self.net = self.create_network(layers, n_channel, omega)
        self.loss_func = self.create_loss_func()

    @abstractmethod
    def create_network(self, layers, omega):
        pass

    @abstractmethod
    def create_loss_func(self):
        pass

    def update_net_params(self, params):
        self.net.net_params = params

    def get_params(self):
        return self.net.net_params

    def forward(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return self.net.f(self.net.net_params, x)

    def gradient(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return self.net.df(self.net.net_params, x)

    def laplace(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        net_out = self.net.d2f(self.net.net_params, x)
        return jnp.sum(net_out, axis = -1)

class NormalImageModel(BaseImageModel):
    def create_network(self, layers, n_channel, omega):
        return Siren(2, layers, n_channel, omega)

    def create_loss_func(self):
        @jit
        def loss_func(net_params, data):
            x = data['input']
            y = data['output']
            output = self.net.f(net_params, x)
            return jnp.mean((output - y)**2)

        return loss_func

class GradientImageModel(BaseImageModel):
    def create_network(self, layers, n_channel, omega):
        if n_channel != 1:
            raise Exception("n_channel should be 1 for GradientImageModel")
        return Siren(2, layers, 1, omega)

    def create_loss_func(self):
        @jit
        def loss_func(net_params, data):
            x = data['input']
            y = data['output']
            output = self.net.df(net_params, x)
            output = output.squeeze(1)
            diff = (output - y)
            return jnp.mean(jnp.sum(diff**2, axis=-1))

        return loss_func

class LaplacianImageModel(GradientImageModel):
    def create_loss_func(self):
        @jit
        def loss_func(net_params, data):
            x = data['input']
            y = data['output']
            output = self.net.d2f(net_params, x)
            laplacian = jnp.sum(output, axis=-1)
            diff = (y - laplacian)
            return jnp.mean(diff**2)

        return loss_func
