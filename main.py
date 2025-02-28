import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

import jax.numpy as jnp
from tqdm import tqdm
# import tqdm.contrib.itertools as itertools
import numpy as np
import argparse
import logging
import optax
import yaml

import matplotlib.pyplot as plt
import itertools
import pickle

from model import *
from utils import MSEmeanloss, TqdmLoggingHandler
from equation import *
from pointgenerate import *

os.makedirs('logs/logs_save', exist_ok = True)
os.makedirs('logs/figure', exist_ok=True)
save_path = 'logs/logs_save'

with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

mode = configure['model']['mode']


if mode == 'nonparameteric':
    forward = nonparaforward
    def model(params):
        return lambda x, y: forward(params, x, y)

elif mode == 'parameteric':
    forward = paraforward
    def model(params):
        return lambda x, y, epsilon, delta, kappa, p: forward(params, x, y, epsilon, delta, kappa, p)
    
elif mode == 'P2INN':
    def model(params):
        return P2INN_pretrain(params)

else:
    raise NotImplementedError

if configure['model']['lrtype'] == 'constant':
    lr = optax.piecewise_constant_schedule(float(configure['model']['init_lr']), {step : float(ler) for step, ler in configure['model']['decay_steps'].items()})
elif configure['model']['lrtype'] == 'cosine':
    lr = optax.cosine_decay_schedule(init_value = configure['model']['init_lr'], decay_steps = configure['model']['decay_steps'])
else:
    raise NotImplementedError

if configure['model']['optimizer'] == 'adam':
    optimizer = optax.adam(lr)
elif configure['model']['optimizer'] == 'sgd':
    optimizer = optax.sgd(lr)
else:
    raise NotImplementedError

def loss(params, bdpts, repts, epsilon, kappa, delta, P = 0.0):
    x_r, y_r =repts[:, 0:1], repts[:, 1:2]

    loss = MSEmeanloss(configure['coeff']['residual'] * gradshafranov(x_r, y_r, model(params), epsilon, kappa, delta, P), jnp.zeros_like(x_r))
    
    x_c, y_c, target_c = bdpts[:, 0:1], bdpts[:, 1:2], bdpts[:, 2:3]

    loss += configure['coeff']['boundary'] * MSEmeanloss(model(params)(x_c, y_c, epsilon, kappa, delta, P), target_c)

    regularization = 0.5 * jnp.sum(jnp.array([jnp.sum(jnp.square(leaves)) for leaves in jax.tree_util.tree_leaves(params)]))
    loss += configure['coeff']['regularizer'] * regularization

    return loss

@jax.jit
def update(opt_state, params, bdpts, repts, epsilon, kappa, delta, pp):
    grads=jax.jit(jax.grad(loss, 0))(params, bdpts, repts, epsilon, kappa, delta, pp)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def main():
    if mode == 'nonparametric':
        epsilon, kappa, delta = configure['data']['domainvariables']
        datas = pointgenerate(
            configure['data']['pointnbrs'],
            configure['data']['domainvariables']
            )(
            boundaryfunction,
            residualfunction
            )

        bdpts, repts = datas['boundary'], datas['residual']
        layers = [2] + configure['model']['layers']
        params = init_params(layers)
        opt_state = optimizer.init(params)

        tqbar = tqdm(range(configure['model']['epoch']))
        log = logging.getLogger(__name__)
        log.handlers = []
        log.setLevel(logging.INFO)
        log.addHandler(TqdmLoggingHandler())

        losses = []
        for epo in tqbar:
            opt_state, params = update(opt_state, params, bdpts, repts)

            if epo % configure['compile_period'] == 0:
                losss = loss(params, bdpts, repts, epsilon, kappa, delta)
                np.save(f"{save_path}/parameters.npy", jax.device_get(params))
                losses.append(losss)
            tqbar.set_postfix({'loss' : losss})
        np.save(f"{save_path}/loss.npy", losses)

        plt.figure(figsize=(15, 15))
        plt.plot(np.arange(configure['model']['epoch'] / configure['compile_period']), losses)
        plt.show()

    elif mode == 'parameteric':
        P_array = jnp.linspace(0.0, 1.0, 5)
        epsilon_array = jnp.linspace(0.12, 0.52, 5)
        kappa_array = jnp.linspace(1.25, 2.75, 5)
        delta_array = jnp.linspace(-0.5 ,0.5, 5)
        losses = []

        tqbar = tqdm(itertools.product(P_array, epsilon_array, kappa_array, delta_array), total = 5 ** 4)
        
        log = logging.getLogger(__name__)
        log.handlers = []
        log.setLevel(logging.INFO)
        log.addHandler(TqdmLoggingHandler())

        for pp, epsilon, kappa, delta in tqbar:
            print(pp, epsilon, kappa, delta)
            datas = pointgenerate(
                configure['data']['pointnbrs'],
                [epsilon, kappa, delta]
                )(
                boundaryfunction,
                residualfunction
                )
            bdpts, repts = datas['boundary'], datas['residual']

            layers = [6] + configure['model']['layers']
            params = init_params(layers)
            
            opt_state = optimizer.init(params)
           
            for epo in range(configure['model']['epoch']):
                opt_state, params = update(opt_state, params, bdpts, repts, epsilon, kappa, delta, pp)

                if epo % configure['compile_period'] == 0:
                    losss = loss(params, bdpts, repts, epsilon, kappa, delta, pp)
                    np.save(f"{save_path}/parameters_{mode}.npy", jax.device_get(params))
                    losses.append(losss)
                    tqbar.set_postfix({'loss' : losss})
                    
        np.save(f"{save_path}/loss_{mode}.npy", losses)
    
        plt.figure(figsize=(15, 15))
        plt.plot(np.arange(len(losses)), losses)
        plt.show()

    elif mode == 'P2INN':
        P_array = jnp.linspace(0.0, 1.0, 5)
        epsilon_array = jnp.linspace(0.12, 0.52, 5)
        kappa_array = jnp.linspace(1.25, 2.75, 5)
        delta_array = jnp.linspace(-0.5 ,0.5, 5)

        losses = []
        P_array = jnp.array([0.0])
        epsilon_array = jnp.array([0.32])
        kappa_array = jnp.array([1.7])
        delta_array = jnp.array([0.33])

        tqbar = tqdm(itertools.product(P_array, epsilon_array, kappa_array, delta_array), total = 1)
        
        log = logging.getLogger(__name__)
        log.handlers = []
        log.setLevel(logging.INFO)
        log.addHandler(TqdmLoggingHandler())

        for pp, epsilon, kappa, delta in tqbar:
            datas = pointgenerate(
                configure['data']['pointnbrs'],
                [epsilon, kappa, delta]
                )(
                boundaryfunction,
                residualfunction
                )
            bdpts, repts = datas['boundary'], datas['residual']

            layers = configure['model']['layers']
            params = init_params_P2INN(layers)
            opt_state = optimizer.init(params)
           
            for epo in tqdm(range(configure['model']['epoch'])):
                opt_state, params = update(opt_state, params, bdpts, repts, epsilon, kappa, delta, pp)
                if epo % configure['compile_period'] == 0:
                    losss = loss(params, bdpts, repts, epsilon, kappa, delta, pp)
                    with open(f"{save_path}/parameters_{mode}.npz", 'wb') as saveparams:
                        pickle.dump(jax.device_get(params), saveparams)
                    losses.append(losss)
                    tqbar.set_postfix({'loss' : losss})
                    
        np.save(f"{save_path}/loss_{mode}.npy", losses)
    
        plt.figure(figsize=(15, 15))
        plt.plot(np.arange(len(losses)), losses)
        plt.ylim(min = 0, max = 1)
        plt.show()

main()