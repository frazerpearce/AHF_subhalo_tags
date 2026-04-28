# periodic-slab

Exact, invertible reparameterisation of a periodic 3D unit cube into a rectangular slab.

This utility maps a cube with periodic boundary conditions onto an
`a × b × (1/(a*b))` slab by changing the fundamental domain of the 3-torus.
The transform is exact, O(1) per point, vectorisable, and invertible.

## Why this exists

This avoids common ad-hoc approaches such as padding, duplication, or interpolation
when working with periodic 3D data.

Typical use cases:
- simulation visualisation
- periodic volume diagnostics
- algorithm testing with periodic domains
- texture / field unwrapping

## Core function

```python
from periodic_slab import cube_to_slab
X, Y, Z = cube_to_slab(x, y, z, a=4, b=2)

