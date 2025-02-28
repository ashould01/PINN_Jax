import jax
import logging
import jax.numpy as jnp
from time import time
from tqdm import tqdm
import sympy as sp
import numpy as np


@jax.jit
def MSEmeanloss(true, target):
    return jnp.mean(jnp.square(true - target))

@jax.jit
def MSEsumloss(true, target):
    return jnp.sum(jnp.square(true - target))
        
def logging_time(func):
    def wrapper_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'Elapsed time for {func.__name__} : {end - start:.3f}')
        return result
    return wrapper_func

class TqdmLoggingHandler(logging.StreamHandler):

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

def true(epsilon, kappa, delta, p):
    x = sp.Symbol('x', real=True, positive=True)
    y = sp.Symbol('y', real = True)

    psip = x ** 4 / 8 + p * (x ** 2 * sp.log(x) / 2 - x ** 4 / 8)

    psi1 = 1

    psi2 = x**2

    psi3 = y**2 - x**2 * sp.log(x)

    psi4 = x**4 - 4*x**2*y**2

    psi5 = 2*y**4 - 9*x**2*y**2 + 3*x**4*sp.log(x) - 12*x**2*y**2*sp.log(x)

    psi6 = x**6 - 12*x**4*y**2 + 8*x**2*y**4

    psi7 = (8*y**6 
            - 140*x**2*y**4 
            + 75*x**4*y**2 
            - 15*x**6*sp.log(x) 
            + 180*x**4*y**2*sp.log(x)
            - 120*x**2*y**4*sp.log(x))

    c1, c2, c3, c4, c5, c6, c7 = sp.symbols('c1 c2 c3 c4 c5 c6 c7', real=True)

    psi = psip + c1 * psi1 + c2 * psi2 + c3 * psi3 + c4 * psi4 + c5 * psi5 + c6 * psi6 + c7 * psi7

    alpha = np.arcsin(delta)

    N1, N2, N3 = - (1 + alpha) ** 2 / (epsilon * kappa ** 2), (1 - alpha) ** 2 / (epsilon * kappa ** 2), - kappa / (epsilon * np.cos(alpha) ** 2)

    eq1 = psi.subs({x : 1 + epsilon, y : 0.0})
    eq2 = psi.subs({x : 1 - epsilon, y : 0.0})
    eq3 = psi.subs({x : 1 - delta * epsilon, y : kappa * epsilon})
    eq4 = psi.diff(x).subs({x : 1 - delta * epsilon, y : kappa * epsilon})
    eq5 = psi.diff(y, y).subs({x : 1 + epsilon, y : 0.0}) + N1 * psi.diff(x).subs({x : 1 + epsilon, y : 0.0})
    eq6 = psi.diff(y, y).subs({x : 1 - epsilon, y : 0.0}) + N2 * psi.diff(x).subs({x : 1 - epsilon, y : 0.0})
    eq7 = psi.diff(x, x).subs({x : 1 - delta * epsilon, y : kappa * epsilon}) + N3 * psi.diff(y).subs({x : 1 - delta * epsilon, y : kappa * epsilon})

    try:
        c = sp.nsolve((eq1, eq2, eq3, eq4, eq5, eq6, eq7), (c1, c2, c3, c4, c5, c6, c7), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), prec = 10)
        cdict = {c1 : c[0], c2 : c[1], c3 : c[2], c4 : c[3], c5 : c[4], c6 : c[5], c7 : c[6]}
        return sp.lambdify((x, y), psi.subs(cdict), modules = 'numpy')
    
    except:
        raise ValueError 