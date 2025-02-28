import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data

X, Y, Z = get_test_data(0.5)

print(f"X.shape={X.shape}")
print(f"Y.shape={Y.shape}")
print(f"Z.shape={Z.shape}")

fig, axs = plt.subplots(ncols=3, figsize=(10, 4), subplot_kw={"projection":"3d"}, constrained_layout=True)

for ax, d in zip(axs, [1, 0.5, 0.1]):
    X, Y, Z = get_test_data(d)
    dim = X.shape[0]
    ax.plot_wireframe(X, Y, Z)
    ax.set_title(f"get_test_data({d}): {dim}x{dim}", fontsize="x-large", color="gray", fontweight="bold")

plt.show()