from scipy.integrate import simps
import numpy as np
y = np.arange(-9,11)
print(simps(y))
x = np.arange(y.size)
print(simps(y,x))

print(simps(y,x))