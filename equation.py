import jax
import jax.numpy as jnp

def gradshafranov(pointr, pointz, psi, epsilon, kappa, delta, P):
    epsilon, kappa, delta, P = epsilon * jnp.ones_like(pointr), kappa * jnp.ones_like(pointr), delta * jnp.ones_like(pointr), P * jnp.ones_like(pointr)
    psi_r = lambda r, z, epsilon, kappa, delta, P: jax.grad(lambda r, z, epsilon, kappa, delta, P: jnp.sum(psi(r, z, epsilon, kappa, delta, P)), 0)(r, z, epsilon, kappa, delta, P)
    psi_rr = lambda r, z, epsilon, kappa, delta, P: jax.grad(lambda r, z, epsilon, kappa, delta, P: jnp.sum(psi_r(r, z, epsilon, kappa, delta, P)), 0)(r, z, epsilon, kappa, delta, P)
    psi_z = lambda r, z, epsilon, kappa, delta, P: jax.grad(lambda r, z, epsilon, kappa, delta, P: jnp.sum(psi(r, z, epsilon, kappa, delta, P)), 1)(r, z, epsilon, kappa, delta, P)
    psi_zz = lambda r, z, epsilon, kappa, delta, P: jax.grad(lambda r, z, epsilon, kappa, delta, P: jnp.sum(psi_z(r, z, epsilon, kappa, delta, P)), 1)(r, z, epsilon, kappa, delta, P)

    residualvalue = psi_rr(pointr, pointz, epsilon, kappa, delta, P) - psi_r(pointr, pointz, epsilon, kappa, delta, P) / pointr + psi_zz(pointr, pointz, epsilon, kappa, delta, P) \
        - P * pointr ** 2 - (1 - P)

    return residualvalue

def boundaryfunction(X):
    return jnp.zeros_like(X[:, 0:1])

def residualfunction(X):
    return jnp.zeros_like(X[:, 0:1])