import jax
import logging
import jax.numpy as jnp
from time import time
from tqdm import tqdm

class pointgenerate:
    
    def __init__(self, pointnbrs, domainvariables):
        
        self.N_b, self.N_r = pointnbrs
        self.epsilon, self.kappa, self.delta = domainvariables

    def radius(self, tau):
        return 1 + self.epsilon * jnp.cos(tau + jnp.arcsin(self.delta) * jnp.sin(tau))

    def boundarypoints(self, boundaryfunction):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
            (boundaryfunction general case = jnp.zeros_like(x))
        '''

        tau = jax.random.uniform(self.boundary_key1, shape = (self.N_b, 1), minval = 0, maxval = jnp.pi * 2)
        r = self.radius(tau)
        z = self.epsilon * self.kappa * jnp.sin(tau)
        x = jnp.concatenate([r, z], axis = 1)

        return jnp.concatenate([r, z, boundaryfunction(x)], axis = 1)
        
    def residualpoints(self, residualfunction):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
        '''
        
        tau = jax.random.uniform(self.residual_key1, shape = (self.N_r, 1), minval = -jnp.pi / 2 , maxval = jnp.pi / 2)
        r_max = self.radius(tau)
        r_min = jnp.where(tau > 0, self.radius(jnp.pi - tau), self.radius(-jnp.pi - tau))
        radius_range = jax.random.uniform(self.residual_key2, shape = (self.N_r, 1), minval = r_min, maxval = r_max)
        z = self.epsilon * self.kappa * jnp.sin(tau)

        residual = jnp.concatenate([radius_range, z], axis = 1)
        r_c = residualfunction(residual)
        
        return jnp.concatenate([radius_range, z, r_c], axis = 1)
    
    def boundarypointsregular(self, pointnumbers):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
            (boundaryfunction general case = jnp.zeros_like(x))
        '''

        tau = jnp.linspace(0, jnp.pi * 2, pointnumbers).reshape(-1, 1)
        r = self.radius(tau)
        z = self.epsilon * self.kappa * jnp.sin(tau)
        x = jnp.concatenate([r, z], axis = 1)

        return x
        
    def residualpointsregular(self, pointnumbers):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
        '''
        
        tau = jnp.linspace(-jnp.pi / 2 , jnp.pi / 2, pointnumbers).reshape(-1, 1)
        r_max = self.radius(tau)
        r_min = jnp.where(tau > 0, self.radius(jnp.pi - tau), self.radius(-jnp.pi - tau))

        radius_range = jnp.linspace(r_min, r_max, pointnumbers)
        z = jnp.array([self.epsilon * self.kappa * jnp.sin(tau).copy() for _ in range(pointnumbers)])

        residual = jnp.concatenate([radius_range, z], axis = 2)
        
        return residual
        
        
    def __call__(self, boundaryfunction, residualfunction):
       
        self.boundary_key1 = jax.random.key(0)
        self.residual_key1, self.residual_key2 = jax.random.split(jax.random.key(0), 2)
        boundary = self.boundarypoints(boundaryfunction)
        residual = self.residualpoints(residualfunction)
        point = {'boundary' : boundary, 'residual' : residual}

        return point
 
