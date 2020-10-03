import numpy as np
import numpy.linalg as la

def esprit(y, number_of_sources):
    m = len(y)
    n = number_of_sources
    
    Y = np.outer(y, y)
    C = np.cov(Y)
    
    U,E,V = la.svd(C, hermitian=True)
    
    S = U[:, :n]
    
    S1 = S[:, 0:m-1]
    S2 = S[:, 1:m]
    
    print(S1.shape)
    print(S2.shape)
    
    P = S1*la.pinv(S2)
    
    # E = eigenvalues, Q = eigenvectors
    L,Q = la.eig(P)
    
    return np.angle(L)