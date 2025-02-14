import jax.numpy as jnp
from tqdm import tqdm
import argparse
import optax
import yaml
import os
from model import *
from utils import pointgenerate, computeloss

parser = argparse.ArgumentParser()

args = parser.parse_args()

os.makedirs('logs/logs_save', exist_ok = True)
save_path = 'logs/logs_save'
with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

class train:
    
    def __init__(self):
        pass

    def model(self, X):
        return forward(self.params, X)
    
    def boundaryfunction(self, X):
        return jnp.zeros_like(X[:, 0:1])
    
    def residualfunction(self, X):
        return jnp.zeros_like(X[:, 0:1])
    
    def loss(self, params, weight = jnp.array([1, 1])):
        datas = pointgenerate(configure['data']['pointnbrs'])(
            configure['data']['domaintype'],
            configure['data']['geometry'],
            [self.boundaryfunction] * 4,
            self.residualfunction
            )
        
        self.loss_output = computeloss(configure['data']['domaintype'], configure['data']['equation'])(
            point = datas,
            params = params,
            model = self.model,
            lossfunction = 'MSE' # utils
        )

        return jnp.inner(weight, self.loss_output)

    @jax.jit
    def __call__(self, params, opt_state):
        # Get the gradient w.r.t to MLP params
        self.params = params
        grads=jax.jit(jax.grad(self.loss, 0))(self.params)

        #Update params
        updates, opt_state = optimizer.update(grads, opt_state)
        self.params = optax.apply_updates(self.params, updates)

        return self.params, opt_state, self.loss_output

if configure['model']['lrtype'] == 'constant':
    lr = optax.constant_schedule(configure['model']['lr'])
# lr = optax.cosine_decay_schedule(init_value = 1e-2, decay_steps = 500)
if configure['model']['optimizer'] == 'adam':
    optimizer = optax.adam(lr)
elif configure['model']['optimizer'] == 'sgd':
    optimizer = optax.sgd(lr)
else:
    raise NotImplementedError

def main():

    jax.profiler.start_trace(save_path)
    params = init_params(configure['model']['layers'])
    for epo in tqdm(range(configure['model']['epoch'])):
        params, opt_state, loss_output = train()(params, opt_state)
        if epo % 100 == 0:
            print(f'Epoch = {epo}, loss = {loss_output} = {jnp.sum(train().loss(params))}', end = '\r')
            with open('logs/logs_save/parameters.txt', 'wb') as file:
                jnp.save(file, params)
    jax.profiler.stop_trace()

    
main()