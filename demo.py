import numpy as np
import matplotlib.pyplot as plt
from periodic_slab import cube_to_slab

# parameters
a, b = 4, 2
n = 50000

rng = np.random.default_rng(1)
x = rng.random(n)
y = rng.random(n)
z = rng.random(n)

X, Y, Z = cube_to_slab(x, y, z, a, b)

# periodic RGB colour to show continuity
rgb = np.stack([
    0.5 * (1 + np.cos(2*np.pi*x)),
    0.5 * (1 + np.cos(2*np.pi*y)),
    0.5 * (1 + np.cos(2*np.pi*z)),
], axis=-1)

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(x, y, z, s=1, c=rgb)
ax1.set_title("Periodic unit cube")
ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.set_zlim(0,1)

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(X, Y, Z, s=1, c=rgb)
ax2.set_title("Unwrapped slab")
ax2.set_xlim(0,a); ax2.set_ylim(0,b); ax2.set_zlim(0,1/(a*b))

plt.tight_layout()
plt.show()
