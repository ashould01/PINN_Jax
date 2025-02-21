import jax.numpy as jnp
from tqdm import tqdm
from time import time
import argparse
import logging
import optax
import yaml
import os
from model import *
from utils import pointgenerate, computeloss, MSEloss

# parser = argparse.ArgumentParser()

# args = parser.parse_args()

os.makedirs('logs/logs_save', exist_ok = True)
save_path = 'logs/logs_save'
with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

def model(params):
    return lambda x, y: forward(params, x, y)

def boundaryleftfunction(X, a = 5):
    xx = X[:, 0:1]
    return xx * (xx ** 2 - 3 * a * xx + 2 * a ** 2)

def boundaryfunction(X):
    return jnp.zeros_like(X[:, 0:1])

def residualfunction(X):
    return jnp.zeros_like(X[:, 0:1])

# def loss(params):
#     datas = pointgenerate(configure['data']['pointnbrs'])(
#         configure['data']['domaintype'],
#         configure['data']['geometry'],
#         [boundaryfunction] * 4,
#         residualfunction
#         )

#     loss_output = computeloss(configure['data']['domaintype'], configure['data']['equation'])(
#         params = params,
#         point = datas,
#         model = model,
#         lossfunction = {'boundary' : 'MSE', 'residual' : 'MSE'} # utils
#     )
#     return jnp.sum(loss_output)

# @jax.jit
# def train(params, opt_state):
#     grads=jax.jit(jax.grad(loss, 0))(params)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state

# if configure['model']['lrtype'] == 'constant':
#     lr = optax.constant_schedule(configure['model']['lr'])
# # lr = optax.cosine_decay_schedule(init_value = 1e-2, decay_steps = 500)
# if configure['model']['optimizer'] == 'adam':
#     optimizer = optax.adam(lr)
# elif configure['model']['optimizer'] == 'sgd':
#     optimizer = optax.sgd(lr)
# else:
#     raise NotImplementedError

class main:
    def __init__(self):
        lr = optax.constant_schedule(1e-2)
        self.optimizer = optax.adam(lr)

    def loss(self, params, bdpts, repts):
        x_r, y_r =repts[:, 0:1], repts[:, 1:2]
    
        loss = MSEloss(computeloss(configure['data']['domaintype'], configure['data']['equation']).laplace(x_r, y_r, model(params)), jnp.zeros_like(x_r)) 

        for bdptidx in range(bdpts.shape[0]):
            x_c, y_c = bdpts[bdptidx, :, 0:1], bdpts[bdptidx, :, 1:2]
            loss += MSEloss(model(params)(x_c, y_c), 0)

        return loss
    

    def update(self, opt_state, params, bdpts, repts):

        grads=jax.grad(self.loss, 0)(params, bdpts, repts)

        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params
    
    def __call__(self):
    
        datas = pointgenerate(configure['data']['pointnbrs'])(
            configure['data']['domaintype'],
            configure['data']['geometry'],
            [boundaryfunction] * 2 + [boundaryleftfunction] + [boundaryfunction],
            residualfunction
            )
        bdpts, repts = datas['boundary'], datas['residual']
        params = init_params(configure['model']['layers'])
        opt_state = self.optimizer.init(params)
        tqbar = tqdm(range(configure['model']['epoch']))



        # log = logging.getLogger(__name__)
        # log.handlers = []
        # log.setLevel(logging.INFO)
        # log.addHandler(TqdmLoggingHandler())

        for epo in tqbar:
            opt_state, params = self.update(opt_state, params, bdpts, repts)
            if epo % 100 == 0:
                # log.info()
                jnp.save('logs/logs_save/parameters.npy', params)


main()()