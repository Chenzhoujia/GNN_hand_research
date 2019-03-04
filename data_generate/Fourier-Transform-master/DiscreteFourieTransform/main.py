import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
from DiscreteFourieTransform.dft import dft
from DiscreteFourieTransform.idft import idft
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

if __name__ == "__main__":
    Fs = 256  # points
    Ts = 1 / 256  # step
    t = np.arange(0, 1, Ts)

    F = 6  # Hz
    y = 8 * sp.sin(2 * sp.pi * F * t)

    n = len(y)
    T = n / Fs
    k = np.arange(n)
    frq = k / T
    frq = frq[range(n // 2)]

    fft = abs(dft(y))
    fft = fft[range(n // 2)]

    ifft = idft(fft)
    n = len(ifft)

    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    ax[0].plot(t, y)
    ax[1].stem(frq, fft, 'r')

    Fs = n
    Ts = 1 / Fs
    t = np.arange(0, 1, Ts)

    ax[2].plot(t, ifft)
    plt.show()
    plt.pause()
