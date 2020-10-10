import numpy as np
import numpy.linalg as la
import numpy.random as rand
from data_gen import generate_los_ula_data


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
    
    print(np.angle(eigs)/(2*np.pi*N*d))
    
    return np.arccos(np.angle(eigs)/(2*np.pi*N*d))


# antennas
N = 5
# users
K = 2
# samples
T = 5
#SNR
SNR = rand.uniform(1,50)
# frequency
f = rand.uniform(2.4e9)

theta,S,Y = generate_los_ula_data(N, K, T, SNR, f)

theta_hat = esprit(Y, N, K, f)

print(theta)
print(theta_hat)