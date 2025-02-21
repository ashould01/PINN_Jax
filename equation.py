import yaml
import jax.numpy as jnp

with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

def boundaryleftfunction(X, a = configure['data']['geometry'][1][0]):
    xx = X[:, 0:1]
    return xx * (xx ** 2 - 3 * a * xx + 2 * a ** 2)

def boundaryfunction(X):
    return jnp.zeros_like(X[:, 0:1])

def residualfunction(X):
    return jnp.zeros_like(X[:, 0:1])