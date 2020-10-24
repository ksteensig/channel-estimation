import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from data_generation import generate_single_data, apply_wgn
from esprit import esprit

# antennas
N = 128

# users
K = 8

# bits
L = 140

# SNR min and max
snr = [5, 30]

# frequency
freq = 1e9

theta, Y = generate_single_data(N, K, L, freq)

theta = theta

Y = apply_wgn(Y, snr)

C = np.cov(Y)

E,U = la.eig(C)

E = np.abs(E)
E[::-1].sort()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True)
plt.semilogy(E)
plt.show()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.plot(E)
plt.show()
print(E[0:10])
