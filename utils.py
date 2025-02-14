import jax
import jax.numpy as jnp

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

        return jnp.concatenate([boundaryleft, boundaryright, boundarydown, boundaryup], axis = 1)
        
    def residualpoints(self, residualfunction):
        
        '''
            input : boundaryfunction
            output : [boundarypoints, boundaryfunctionvalue] -> shape (N_b, 3)
        '''
        
        x_c = jax.random.uniform(self.residual_key1, minval = self.xmin, maxval = self.xmax)
        y_c = jax.random.uniform(self.residual_key2, minval = self.ymin, maxval = self.ymax)
        residual = jnp.concatenate([x_c, y_c], axis = 1)
        r_c = residualfunction(residual)
        
        return jnp.concatenate([x_c, y_c, r_c], axis = 1)
        
        
    def __main__(self, domaintype, geometry, boundaryfunction, residualfunction):
        
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
            raise NotImplementedError
        
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
        
    def MSEloss(self, true, target, mode = 'mean'):
        if mode == 'mean':
            return jnp.mean(jnp.square(true - target))
        elif mode == 'sum':
            return jnp.sum(jnp.square(true - target))
        
    
    def laplace(self, params, point, u):
        u_x = jax.grad(u(params, point), 1)
        u_xx = jax.grad(u_x(params, point), 1)
        u_y = jax.grad(u(params, point), 2)
        u_yy = jax.grad(u_y(params, point), 2)
        
        return (u_xx + u_yy).reshape(-1, 1)
    
    def __call__(self, point, params, model, lossfunction):
        
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
                boundarymodelpoint = point['boundary'][j, :, 0:2]
                boundarytargetpoint = point['boundary'][j, :, 2:3]
                boundaryloss += boundarylossftn(model(params, boundarymodelpoint), boundarytargetpoint)

            residualmodelpoint = point['residual'][:, 0:2]
            residualtargetpoint = point['residual'][:, 2:3]
            residualloss = residuallossftn(equationfunction(params, boundarytargetpoint, model(params, residualmodelpoint)), residualmodelpoint)

            return [boundaryloss, residualloss]
        
        else:
            raise NotImplementedError
            
        
