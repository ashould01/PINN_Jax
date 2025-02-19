import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def init_params(layers):
    
    '''
        input = int list
        output = parameters
    '''
    
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers)-1)
    params = list()
    initializer = jax.nn.initializers.glorot_normal()
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = initializer(key, (n_in, n_out), jnp.float32)
        B = jax.random.uniform(key, shape=(n_out,), dtype = jnp.float32)
        params.append({'W' : W,'B' : B})
    return params

def forward(params, X):
    
    '''
        input = parameters, residual points
        output = model output
    '''
    

    *hidden, last = params
    for layer in hidden :
        X = jax.nn.tanh(X @ layer['W']+layer['B'])
    return X @ last['W'] + last['B']