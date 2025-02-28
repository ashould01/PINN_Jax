import matplotlib.pyplot as plt
import matplotlib
import jax.numpy as jnp
import numpy as np
import yaml
import pickle

from pointgenerate import pointgenerate
from model import *
from utils import true

with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

epsilon, kappa, delta, P = 0.32, 1.7, 0.33, 0.0
boundary = pointgenerate([1, 1], [epsilon, kappa, delta]).boundarypointsregular(101)

Rboundary, Zboundary = boundary[:, 0:1], boundary[:, 1:2]

residual = pointgenerate([1, 1], [epsilon, kappa, delta]).residualpointsregular(101)
Rresidual, Zresidual = residual[:, :, 0:1], residual[:, :, 1:2]

if configure['model']['mode'] == 'parameteric':
    with open('logs/logs_save/parameters_param.npy', 'rb') as file:
        params = jnp.load(file, allow_pickle = True)
    pred_r = paraforward(params, Rresidual.flatten().reshape(-1, 1), Zresidual.flatten().reshape(-1, 1), epsilon, kappa, delta, P).reshape(Rresidual.shape)
elif configure['model']['mode'] == 'P2INN':
    with open('logs/logs_save/parameters_P2INN.npz', 'rb') as file:
        params = pickle.load(file)
        pred_r = P2INN_pretrain(dict(params))(Rresidual.flatten().reshape(-1, 1), Zresidual.flatten().reshape(-1, 1), epsilon, kappa, delta, P).reshape(Rresidual.shape)

true_r = true(epsilon, kappa, delta, P)(Rresidual, Zresidual).squeeze()

# print(f'loss : {MSEmeanloss(pred, truev)}')

fig = plt.figure(figsize = (30, 15))
plt.rc('font', size = 20)
ax1 = fig.add_subplot(121)
ax1.set_title('True')

# ax.pcolormesh(Rresidual.squeeze(), Zresidual.squeeze(), pred_r.squeeze())
# ax.pcolormesh(Rresidual.squeeze(), Zresidual.squeeze(), )
cmap_1 = matplotlib.colormaps.get_cmap("Spectral")
vmax_1 = jnp.max(true_r)
vmin_1 = jnp.min(true_r)
norm_1 = matplotlib.colors.Normalize(vmin = vmin_1, vmax = vmax_1)
colormapping_1 = matplotlib.cm.ScalarMappable(norm = norm_1, cmap = cmap_1)

ax1.contourf(Rresidual.squeeze(), Zresidual.squeeze(), true_r, levels = 25, cmap = cmap_1)
contour = ax1.contour(Rresidual.squeeze(), Zresidual.squeeze(), true(epsilon, kappa, delta, P)(Rresidual.squeeze(), Zresidual.squeeze()), levels = 25, colors = 'white')
ax1.plot(Rboundary, Zboundary, color = 'black')
ax1.clabel(contour, inline=True, inline_spacing=15, fontsize=20)
fig.colorbar(colormapping_1, ax = ax1)
ax1.set_xlabel('R')
ax1.set_ylabel('Z')

ax2 = fig.add_subplot(122)
ax2.set_title('Predict')

cmap_2 = matplotlib.colormaps.get_cmap("Spectral")
vmax_2 = jnp.max(pred_r)
vmin_2 = jnp.min(pred_r)
norm_2 = matplotlib.colors.Normalize(vmin = vmin_2, vmax = vmax_2)
colormapping_2 = matplotlib.cm.ScalarMappable(norm = norm_2, cmap = cmap_2)

ax2.contourf(Rresidual.squeeze(), Zresidual.squeeze(), pred_r.squeeze(), levels = 25, cmap = cmap_2)
contour = ax2.contour(Rresidual.squeeze(), Zresidual.squeeze(), pred_r.squeeze(), levels = 25, colors = 'white')
ax2.plot(Rboundary, Zboundary, color = 'black')
ax2.clabel(contour, inline=True, inline_spacing=15, fontsize=20)
fig.colorbar(colormapping_2, ax = ax2)
ax2.set_xlabel('R')
ax2.set_ylabel('Z')
fig.suptitle(f'Grad-Shafranov equation, epsilon={epsilon}, kappa={kappa}, delta={delta}, P={P}')

plt.savefig(f'logs/figure/ep_{epsilon}-kappa_{kappa}-delta_{delta}-P_{P}.png')
plt.show()