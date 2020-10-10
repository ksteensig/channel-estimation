import numpy as np
import numpy.random as rand
#import scipy.io

# needs an SNR parameter, to compute the noise variance
def generate_los_ula_data(N, K, T, SNR, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    theta = rand.uniform(0, np.pi, (K,1))
    
    alpha = rand.randn(K,2).view(np.complex128)
    
    H = np.zeros((N,K), dtype = 'complex_')

    spatial_const = 2j*np.pi*d/wl
    
    for n in range(N):
        for k in range(K):
            H[n,k] = alpha[k]*np.exp(spatial_const*np.cos(theta[k]))
            
    Signal = SNR*rand.randn(K,T)
    
    Noise = rand.randn(N,T*2).view(np.complex128)
    
    # y = Hs + n
    Received = H.dot(Signal) + Noise
    
    return theta,Signal,Received