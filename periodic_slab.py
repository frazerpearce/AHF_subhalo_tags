import numpy as np


def cube_to_slab(x, y, z, a, b):
    """
    Map periodic unit-cube coords (x,y,z) in [0,1) to an a×b slab
    of depth 1/(a*b).

    Output:
      X in [0,a), Y in [0,b), Z in [0,1/(a*b))

    Planes: (z - x/a - y/b) = const (mod 1)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive integers")

    ab = a * b

    r = (z - x / a - y / b) % 1.0
    n = np.floor(ab * r).astype(np.int64)

    ia = n % a
    ib = n // a

    X = x + ia
    Y = y + ib
    Z = r - n / ab

    return X, Y, Z
