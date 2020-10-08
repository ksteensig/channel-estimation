import numpy as np
import numpy.linalg as la
import numpy.random as rand
#import scipy.io

# needs an SNR parameter, to compute the noise variance
def generate_los_ula_data(N, K, T, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    theta = rand.uniform(0, np.pi, (K,1))
    
    # make this complex normal distributed
    alpha = rand.uniform(size=(K,1))
    
    H = np.zeros((N,K), dtype = 'complex_')

    spatial_const = 2j*np.pi*d/wl
    
    for n in range(N):
        for k in range(K):
            angle = theta[k]
            attenuation = alpha[k]
            H[n,k] = attenuation*np.exp(spatial_const*np.cos(angle))
            
    S = 100*np.ones((K,T)) #rand.randn(K,5)
    
    # should be complex normal
    N = rand.randn(N,T)
    
    # y = Hs + n
    Y = H.dot(S) + N
    
    return theta,S,Y


def esprit(Y, N, K, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    m = len(Y)
    
    # \hat R = E[Y*Y^H] = 1/m * Y*Y^H
    R = Y.dot(Y.conj().T) / m
    
    # ensure matrix is hermitian
    #R = 0.5 * (R + R.conj().T)
    
    # ensure descending order of U, E, V
    U,E,V = la.svd(R)
    
    S = U[:, 0:K]
    
    S1 = S[0:m-1,:]
    S2 = S[1:m,:]
    
    # P = S1 \ S2
    P = la.pinv(S1) @ S2
    
    # E = eigenvalues, Q = eigenvectors
    eigs,_ = la.eig(P)
    
    return np.arccos(np.angle(eigs)/(2*np.pi*N*d))


# antennas
N = 5
# users
K = 2
# samples
T = 5
# frequency
f = 1e9

theta,S,Y = generate_los_ula_data(N, K, T, f)

theta_hat = esprit(Y, N, K, f)

print(theta)
print(theta_hat)