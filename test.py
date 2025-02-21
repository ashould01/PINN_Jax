import matplotlib.pyplot as plt
import jax.numpy as jnp
from model import forward
import yaml
import numpy as np
from utils import MSEmeanloss

with open('config.yaml', 'rb') as f:
    configure = yaml.safe_load(f)

with open('logs/logs_save/parameters.npy', 'rb') as file:
    params = jnp.load(file, allow_pickle = True)

xmin, ymin, xmax, ymax = np.array(configure['data']['geometry']).flatten()

pi = np.pi
def true(x, y):
    ret = 0
    for i in range(1, 10):
        ret += np.sinh(i * pi * (ymax - y) / xmax) / (i ** 3 * np.sinh(i * pi * ymax / xmax)) * np.sin(i * pi * x / xmax)
    return ret * 12*2**3 / np.pi**3

x, y = jnp.linspace(xmin, xmax, 1001), jnp.linspace(ymin, ymax, 1001)
xx, yy = jnp.meshgrid(x, y)

pred = forward(params, xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)).reshape(xx.shape)
truev = true(xx, yy)

print(f'loss : {MSEmeanloss(pred, truev)}')

fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(projection = '3d')

ax.plot_surface(xx, yy, pred)
ax.plot_surface(xx, yy, truev, alpha = 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()