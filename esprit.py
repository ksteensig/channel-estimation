import numpy as np
import numpy.linalg as la
import numpy.random as rand
import scipy.io

def generate_los_ula_data(N, K, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    theta = rand.uniform(0, np.pi, (K,1))
    alpha = 3*rand.uniform(size=(K,1))
    
    H = np.zeros((N,K), dtype = 'complex_')

    spatial_const = 2j*np.pi*d/wl
    
    for n in range(N):
        for k in range(K):
            angle = theta[k]
            attenuation = alpha[k]
            H[n,k] = attenuation*np.exp(spatial_const*np.sin(angle))
            
    s = rand.randn(K,1)
    n = rand.randn(N,1)
    
    # y = Hs + n
    y = H.dot(s) + n
    
    return theta,s,y
            
    

def esprit(y, N, K):
    m = len(y)
    
    R = np.outer(y, y)
    
    U,E,V = la.svd(R)
    
    S = U[:, 0:K]
    
    S1 = S[0:m-1]
    S2 = S[1:m]
    
    P = la.pinv(S1) @ S2
    
    # E = eigenvalues, Q = eigenvectors
    eigs,_ = la.eig(P)
    
    return np.arcsin(np.angle(eigs)/np.pi)


#x = np.array([np.random.randn(6)]).transpose()

#a = esprit(x, 2, 1)

theta,s,y = generate_los_ula_data(5, 2, 1e9)

scipy.io.savemat('test.mat', dict(y=y, theta=theta, s=s))

theta_hat = esprit(y, 5, 2)

print(theta)
print(theta_hat)