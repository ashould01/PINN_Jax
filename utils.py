import jax
import jax.numpy as jnp
from time import time
from tqdm import tqdm

class pointgenerate:
    
    def __init__(self, pointnbrs):
        
        '''
        boundary conditions
        pointnbrs = [N_i, N_b, N_r]
        geometry = {rectangle : [[xmin, ymin], [xmax, ymax]]}
        '''
        self.N_b, self.N_r = pointnbrs
        # self.N_i, self.N_b, self.N_r= pointnbrs


    def rectboundarypoints(self, boundaryfunction):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (4 (edges), N_b, 3)
        '''

        # (xmin, y)
        y_b1 = jax.random.uniform(self.boundary_key1, minval = self.ymin, maxval = self.ymax, shape=(self.N_b, 1))
        x_b1 = jnp.ones_like(y_b1) * self.xmin
        boundary1 = jnp.concatenate([x_b1, y_b1], axis = 1)
        bc_1 = boundaryfunction[0](boundary1)
        boundaryleft = jnp.concatenate([boundary1, bc_1], axis = 1)

        # (xmax, y)
        y_b2 = jax.random.uniform(self.boundary_key2, minval = self.ymin, maxval = self.ymax, shape=(self.N_b, 1))
        x_b2 = jnp.ones_like(y_b2) * self.xmax
        boundary2 = jnp.concatenate([x_b2, y_b2], axis = 1)
        bc_2 = boundaryfunction[1](boundary2)
        boundaryright = jnp.concatenate([boundary2, bc_2], axis=1)

        # (x, ymin)
        x_b3 = jax.random.uniform(self.boundary_key3, minval = self.xmin, maxval = self.xmax, shape=(self.N_b, 1))
        y_b3 = jnp.ones_like(x_b3) * self.ymin
        boundary3 = jnp.concatenate([x_b3, y_b3], axis = 1)
        bc_3 = boundaryfunction[2](boundary3)
        boundarydown = jnp.concatenate([boundary3, bc_3], axis=1)

        # (x, ymax)
        x_b4 = jax.random.uniform(self.boundary_key4, minval = self.xmin, maxval = self.xmax, shape=(self.N_b, 1))
        y_b4 = jnp.ones_like(x_b4) * self.ymax
        boundary4 = jnp.concatenate([x_b4, y_b4], axis = 1)
        bc_4 = boundaryfunction[3](boundary4)
        boundaryup = jnp.concatenate([boundary4, bc_4], axis=1)

        return jnp.concatenate([boundaryleft[None, :, :], boundaryright[None, :, :], boundarydown[None, :, :], boundaryup[None, :, :]], axis = 0)
        
    def residualpoints(self, residualfunction):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
        '''
        
        x_c = jax.random.uniform(self.residual_key1, minval = self.xmin, maxval = self.xmax).reshape(-1, 1)
        y_c = jax.random.uniform(self.residual_key2, minval = self.ymin, maxval = self.ymax).reshape(-1, 1)
        residual = jnp.concatenate([x_c, y_c], axis = 1)
        r_c = residualfunction(residual)
        
        return jnp.concatenate([x_c, y_c, r_c], axis = 1)
        
        
    def __call__(self, domaintype, geometry, boundaryfunction, residualfunction):
        
        if domaintype == 'rectangular':
            self.boundary_key1, self.boundary_key2, self.boundary_key3, self.boundary_key4 = jax.random.split(jax.random.PRNGKey(0), 4)
            self.residual_key1, self.residual_key2 = jax.random.split(jax.random.PRNGKey(0), 2)
            self.xmin, self.ymin = geometry[0]
            self.xmax, self.ymax = geometry[1]
            # initial = self.initialpoints()
            boundary = self.rectboundarypoints(boundaryfunction)
            residual = self.residualpoints(residualfunction)
            point = {'boundary' : boundary, 'residual' : residual}
            
            return point
        else:
            return NotImplementedError

@jax.jit
def MSEloss(true, target, mode = 'mean'):
    if mode == 'mean':
        return jnp.mean(jnp.square(true - target))
    elif mode == 'sum':
        return jnp.sum(jnp.square(true - target))

class computeloss:

    def __init__(self, domaintype, equation):
        self.domaintype = domaintype
        self.equation = equation
        if equation in ['burgers']:
            self.timemode = True
        
        elif equation in ['laplace']:
            self.timemode = False
        
        else:
            raise NotImplementedError
    
    def laplace(self, pointx, pointy, u):
        u_x = jax.grad(lambda x, y: jnp.sum(u(x, y)), 0)
        u_xx = jax.grad(lambda x, y: jnp.sum(u_x(x, y)), 0)
        u_y = jax.grad(lambda x, y: jnp.sum(u(x, y)), 1)
        u_yy = jax.grad(lambda x, y: jnp.sum(u_y(x, y)), 1)
        
        return u_xx(pointx, pointy) + u_yy(pointx, pointy)
    
    def __call__(self, params, point, model, lossfunction):
        
        if self.equation == 'laplace':
            equationfunction = self.laplace
        else:
            NotImplementedError
        # if lossfunction['initial'] == 'L2':
        #     initloss = self.L2loss
        # else:
        #     raise NotImplementedError
        
        if lossfunction['boundary'] == 'MSE':
            boundarylossftn = self.MSEloss
        else:
            raise NotImplementedError
        
        if lossfunction['residual'] == 'MSE':
            residuallossftn = self.MSEloss
        else:
            raise NotImplementedError
        
        if self.domaintype == 'rectangular':
            boundaryloss = 0
            for j in range(point['boundary'].shape[0]):
                boundarymodelpointx, boundarymodelpointy = point['boundary'][j, :, 0:1], point['boundary'][j, :, 1:2]  
                boundarytargetpoint = point['boundary'][j, :, 2:3]
                boundaryloss += boundarylossftn(model(params)(boundarymodelpointx, boundarymodelpointy), boundarytargetpoint)

            residualmodelpointx, residualmodelpointy = point['residual'][:, 0:1], point['residual'][:, 1:2]
            residualtargetpoint = point['residual'][:, 2:3]
            residualloss = residuallossftn(equationfunction(residualmodelpointx, residualmodelpointy, model(params)), residualtargetpoint)

            return jnp.array([boundaryloss, residualloss])
        
        else:
            raise NotImplementedError
        
def logging_time(func):
    def wrapper_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'Elapsed time for {func.__name__} : {end - start:.3f}')
        return result
    return wrapper_func

# class TqdmLoggingHandler(logging.StreamHandler):
#     """Avoid tqdm progress bar interruption by logger's output to console"""
#     # see logging.StreamHandler.eval method:
#     # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
#     # and tqdm.write method:
#     # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg, end=self.terminator)
#         except RecursionError:
#             raise
#         except Exception:
#             self.handleError(record)