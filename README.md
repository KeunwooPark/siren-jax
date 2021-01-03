# Siren-Jax
Unofficial implementation of Siren with Jax. This code reproduces image-related results in the original Siren papaer.

## What is Siren?
It is a novel neural network that is proposed in the **Implicit Neural Representations
with Periodic Activation Functions** by Sitzmann et al. 

Siren uses sine functions as activation functions and it can represent continous differentiable signals bettern than networks with other activation functions.

If you want to know more about Siren, please check out the [project page](https://vsitzmann.github.io/siren/)

## How to use?

### 1. Install Jax
Please follow the [official install guide](https://github.com/google/jax).

### 2. Install packages
```shell
$ pip install -r requirements.txt
```

### 3. Train
This code runs the default training option. Please run ```python train.py -h``` to see other options.
```shell
$ python train.py --file reference.jpg
```

### 4. Test
```shell
$ python test.py --run_name reference
```

## Example Results
This section shows results of *implicite image representation* and *solving Possion equation*.

### Reproducing Paper Results


