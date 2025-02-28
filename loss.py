import jax.numpy as jnp

from model import paraforward
from utils import true, MSEmeanloss
from pointgenerate import pointgenerate


def absloss(R, Z, epsilon, kappa, delta, P):
    params = jnp.load('logs/logs_save/parameters_param.npy', allow_pickle = True)
    estimate = paraforward(params, R, Z, epsilon, kappa, delta, P)
    truepsi = true(epsilon, kappa, delta, P)(R, Z)

    return MSEmeanloss(estimate, truepsi)

def relloss(R, Z, epsilon, kappa, delta, P):
    params = jnp.load('logs/logs_save/parameters_param.npy', allow_pickle = True)
    estimate = paraforward(params, R, Z, epsilon, kappa, delta, P)
    truepsi = true(epsilon, kappa, delta, P)(R, Z)

    return MSEmeanloss(estimate, truepsi)/MSEmeanloss(truepsi, jnp.zeros_like(truepsi))

epsilon, kappa, P = 0.52, 2.75, 1.0

for delta in jnp.linspace(-0.5, 0.5, 5):

    boundary = pointgenerate([1, 1], [epsilon, kappa, delta]).boundarypointsregular(101)

    Rboundary, Zboundary = boundary[:, 0:1], boundary[:, 1:2]

    residual = pointgenerate([1, 1], [epsilon, kappa, delta]).residualpointsregular(101)
    Rresidual, Zresidual = residual[:, :, 0:1], residual[:, :, 1:2]

    print(absloss(Rresidual.flatten().reshape(-1, 1), Zresidual.flatten().reshape(-1, 1), epsilon, kappa, delta, P))
    print(relloss(Rresidual.flatten().reshape(-1, 1), Zresidual.flatten().reshape(-1, 1), epsilon, kappa, delta, P))