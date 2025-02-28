import jax
import jax.numpy as jnp
import yaml

with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

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
        params.append({'W' : W, 'B' : B})
    return params

def init_params_P2INN(layers):

    '''
        input = dictionary; {encodereq : list, encoderpara : list, decoder : list}
        output = parameters
    '''
    assert set(layers.keys()) == set(['encodereq', 'encoderpara', 'decoder']), f'{layers.keys()}'

    keys = {}
    keys['encodereq'], keys['encoderpara'], keys['decoder'] = \
        jax.random.split(jax.random.key(0), len(layers['encodereq']) - 1), \
        jax.random.split(jax.random.key(0), len(layers['encoderpara']) - 1), \
        jax.random.split(jax.random.key(0), len(layers['decoder']) - 1)
    
    initializer = jax.nn.initializers.glorot_normal()

    params = {}

    for modeltype in layers.keys():
        params[modeltype] = []
        for key, n_in, n_out in zip(keys[modeltype], layers[modeltype][:-1], layers[modeltype][1:]):
            W = initializer(key, (n_in, n_out), jnp.float32)
            B = jax.random.uniform(key, shape=(n_out,), dtype = jnp.float32)
            params[modeltype].append({'W' : W, 'B' : B})
    return params

    
def nonparaforward(params, x, y):
    '''
        input = parameters, residual points
        output = model output
    '''
    X = jnp.concatenate([x, y], axis = 1)
    *hidden, last = params
    for layer in hidden :
        X = jax.nn.tanh(X @ layer['W']+layer['B'])
    return X @ last['W'] + last['B']

def paraforward(params, x, y, epsilon, kappa, delta, p):
    '''
        input = parameters, residual points
        output = model output
    '''
    epsilon, kappa, delta, p = epsilon * jnp.ones_like(x), kappa * jnp.ones_like(x), delta * jnp.ones_like(x), p * jnp.ones_like(x)
    X = jnp.concatenate([x, y, epsilon, delta, kappa, p], axis = 1)

    *hidden, last = params
    for layer in hidden :
        

        X = jax.nn.tanh(X @ layer['W']+layer['B'])
    return X @ last['W'] + last['B']

class P2INN_pretrain():

    def __init__(self, params):
        self.paramsencodereq = params['encodereq']
        self.paramsencoderpara = params['encoderpara']
        self.paramsdecoder = params['decoder']

    def p2innencodereq(self, params, x, y):

        X = jnp.concatenate([x, y], axis = 1)
        *hidden, last = params
        for layer in hidden:
            X = jax.nn.tanh(X @ layer['W']+layer['B'])
        return X @ last['W'] + last['B']

    def p2innencoderpara(self, params, epsilon, kappa, delta, p):

        X = jnp.concatenate([epsilon, kappa, delta, p], axis = 1)
        *hidden, last = params
        for layer in hidden:
            X = jax.nn.tanh(X @ layer['W']+layer['B'])
        return X @ last['W'] + last['B']
    
    def p2inndecoder(self, params, latent1, latent2):

        X = jnp.concatenate([latent1, latent2], axis = 1)
        *hidden, last = params
        for layer in hidden:
            X = jax.nn.relu(X @ layer['W']+layer['B'])
        return X @ last['W'] + last['B']

    
    def __call__(self, x, y, epsilon, kappa, delta, p):

        epsilon, kappa, delta, p = jnp.ones_like(x) * epsilon, jnp.ones_like(x) * kappa, jnp.ones_like(x) * delta, jnp.ones_like(x) * p
        latent1 = self.p2innencodereq(self.paramsencodereq, x, y)
        latent2 = self.p2innencoderpara(self.paramsencoderpara, epsilon, kappa, delta, p)
        u = self.p2inndecoder(self.paramsdecoder, latent1, latent2)

        return u


