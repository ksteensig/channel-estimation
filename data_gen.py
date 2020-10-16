import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt

# needs an SNR parameter, to compute the noise variance
def generate_los_ula_data(N, K, T, SNR, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    theta = rand.uniform(0, np.pi, (K,1))
    
    alpha = rand.randn(K,2).view(np.complex128)
    
    H = np.zeros((N,K), dtype = 'complex_')

    spatial_const = -2j*np.pi*d/wl
    
    for k in range(K):
        for n in range(N):
            H[n,k] = alpha[k]*np.exp(spatial_const*n*np.cos(theta[k]))
    
    Signal = np.ones((K,T))
    
    Noise = rand.randn(N,T*2).view(np.complex128)
    
    # y = Hs + n
    Received = H.dot(Signal) + Noise/3
    
    return theta,Signal,Received

"""
# antennas
N = 32

# users
K = 8

# frequency
f = 1e9

SNR = 10

training = []
labels = []

for i in range(200):
    label,_,train = generate_los_ula_data(N, K, 1, SNR, f)
    training.append(train.T)
    labels.append(label.T)
    
training = np.array(training)
labels = np.array(labels)

C = np.cov(training.reshape(200,N).T)

E,U = la.eig(C)

plt.plot(np.abs(E))
"""