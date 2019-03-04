import numpy as np
import scipy as sp


def idft(f):
    n = f.size
    result = np.zeros(n, dtype=complex)
    for u in range(n):
        z = 0
        for x in range(n):
            p = 2 * sp.pi * u * x / n
            z += f[x] * (sp.cos(p) + sp.sin(p) * 1j)
        result[u] = z / np.sqrt(n)
    return result
