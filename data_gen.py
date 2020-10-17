import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt

def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)

# needs an SNR parameter, to compute the noise variance
def generate_los_ula_data(N, K, T, SNR, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    theta = rand.uniform(-np.pi/2, np.pi/2, (K))
    
    array = np.linspace(0,N-1,N)*d/wl

    alpha = np.random.randn(K) + np.random.randn(K)*1j # random source powers
    alpha = np.sqrt(1/2)*alpha

    H = np.zeros((N,T)) + 1j*np.zeros((N,T))

    for iter in range(T):
        htmp = np.zeros(N)
        for i in range(K):
            pha = np.exp(1j*2*np.pi*np.random.rand(1))
            htmp = htmp + alpha[i]*pha*array_response_vector(array,theta[i])
        H[:,iter] = htmp + np.sqrt(0.5/SNR)*(np.random.randn(N)+np.random.randn(N)*1j)

    theta = np.repeat(theta, T).reshape(K,T).T

    return np.sort(theta),H

def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = la.eig(CovMat)
    S = U[:,0:L]
    Phi = la.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = la.eig(Phi)
    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
    return np.sort(DoAsESPRIT)

"""
# antennas
N = 128

# users
K = 16

# bits
T = 16

# frequency
f = 1e9

SNR = 30

#np.random.seed(1)

theta,H = generate_los_ula_data(N, K, T, SNR, f)

theta = theta[0]

print(theta)

CovMat = H@H.conj().transpose()

theta_hat = np.sort(esprit(CovMat, K, N))

print(theta_hat)

mse = ((theta - theta_hat)**2).mean()

print(mse)

E,U = la.eig(CovMat)

plt.plot(np.abs(E))
plt.show()
"""
