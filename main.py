import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import argparse
import logging
import optax
import yaml
import os
import matplotlib.pyplot as plt

from equation import *
from model import *
from utils import MSEmeanloss, TqdmLoggingHandler, pointgenerate, computeloss

os.makedirs('logs/logs_save', exist_ok = True)
save_path = 'logs/logs_save'
with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

# parser = argparse.ArgumentParser()

# args = parser.parse_args()

def model(params):
    return lambda x, y: forward(params, x, y)

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

def loss(params, bdpts, repts):
    x_r, y_r =repts[:, 0:1], repts[:, 1:2]

    loss = configure['coeff']['residual'] * MSEmeanloss(computeloss(configure['data']['domaintype'], configure['data']['equation']).laplace(x_r, y_r, model(params)), jnp.zeros_like(x_r)) 
    for bdptidx in range(bdpts.shape[0]):
        x_c, y_c, target_c = bdpts[bdptidx, :, 0:1], bdpts[bdptidx, :, 1:2], bdpts[bdptidx, :, 2:3]
        loss += configure['coeff']['boundary'] * MSEmeanloss(model(params)(x_c, y_c), target_c)

    regularization = 0.5 * jnp.sum(jnp.array([jnp.sum(jnp.square(leaves)) for leaves in jax.tree_util.tree_leaves(params)]))
    loss += configure['coeff']['regularizer'] * regularization

    return loss

@jax.jit
def update(opt_state, params, bdpts, repts):
    grads=jax.jit(jax.grad(loss, 0))(params, bdpts, repts)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def main():

    datas = pointgenerate(configure['data']['pointnbrs'])(
        configure['data']['domaintype'],
        configure['data']['geometry'],
        [boundaryfunction] * 2 + [boundaryleftfunction] + [boundaryfunction],
        residualfunction
        )
    
    bdpts, repts = datas['boundary'], datas['residual']
    params = init_params(configure['model']['layers'])
    opt_state = optimizer.init(params)

    tqbar = tqdm(range(configure['model']['epoch']))
    log = logging.getLogger(__name__)
    log.handlers = []
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())

    losses = []
    for epo in tqbar:
        opt_state, params = update(opt_state, params, bdpts, repts)
        if epo % 1000 == 0:
            losss = loss(params, bdpts, repts)
            log.info(f"loss : {losss}")
            np.save(f"{save_path}/parameters.npy", jax.device_get(params))
            losses.append(losss)
    
    plt.plot(configure['model']['epoch'], losses)

main()