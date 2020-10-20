import numpy as np
import numpy.random as rand
import numpy.linalg as la


# SNR is a range between min and max SNR in dB
def generate_los_ula_data(N, K, L, SNR, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas

    # choose random snr
    db2pow = 10**(rand.uniform(SNR[0], SNR[1])/10)
    
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl
    
    # steering vector
    array_response = lambda array,theta: np.exp(-1j*2*np.pi*array*np.cos(theta))

    theta = np.sort(np.pi*np.random.rand(K))
    alpha = (np.random.randn(K,L) + 1j*np.random.randn(K,L))*np.sqrt(1/2)
    
    x = np.exp(1j*2*np.pi*np.random.rand(K))
    
    Y = np.zeros((N,L)) + 1j*np.zeros((N,L))
        
    # assume the source vector x is a vector containing ones
    # i.e. x = [1 1 ... 1]^T
    for l in range(L):
        for k in range(K):
            response = array_response(array, theta[k])
            Y[:,l] += alpha[k,l]*response*x[k]#*np.exp(-1j*2*np.pi*np.random.rand(1))
    # N = [n1 n2 .. nL]
    N = (rand.randn(N,L) + 1j*rand.randn(N,L))*np.sqrt(0.5/db2pow)

    # y = Hx + n
    Y += N
        
    return theta, Y

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# antennas
N = 128

# users
K = 8

# bits
L = 100

# SNR min and max
snr = [20, 20]

# frequency
f = 1e9

theta, Y = generate_los_ula_data(N, K, L, snr, f)

C = np.cov(Y)

E,U = la.eig(C)

E = np.abs(E)
E[::-1].sort()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True)
#plt.yscale('log')
plt.semilogy(E)
plt.show()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.plot(E)
plt.show()
print(E[0:10])
"""