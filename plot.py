import matplotlib.pyplot as plt
import jax.numpy as jnp
from model import forward
import numpy as np

with open('logs/logs_save/parameters.npy', 'rb') as file:
    params = jnp.load(file, allow_pickle = True)

fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(projection = '3d')
x, y = jnp.linspace(-1, 1, 101), jnp.linspace(-1, 1, 101)
xx, yy = jnp.meshgrid(x, y)

ax.plot_surface(xx, yy, forward(params, xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)).reshape(xx.shape))

plt.show()