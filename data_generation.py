import numpy as np
import numpy.random as rand
from os.path import isfile

# SNR is a range between min and max SNR in dB
def generate_single_data(N, K, L, f):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas    
    
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl
    
    # steering vector
    array_response = lambda array,theta: np.exp(-1j*2*np.pi*array*np.cos(theta))

    theta = np.sort(np.pi*np.random.rand(K,1), axis=0)
    alpha = (np.random.randn(K) + 1j*np.random.randn(K))*np.sqrt(1/2)
    
    # random inputs with |x_{k,l}| = 1
    x = np.exp(-1j*2*np.pi*np.random.rand(K, L))
    
    # realizations of received data, with L being number of realizations
    # Y = [y1 y2 .. yL]
    Y = np.zeros((N,L)) + 1j*np.zeros((N,L))
    
    for l in range(L):
        for k in range(K):
            response = array_response(array, theta[k])
            Y[:,l] += alpha[k]*response*x[k,l]
        
    return np.repeat(theta, L, axis=1).T, Y

def apply_wgn(Y, SNR):
    shape = Y.shape
    db2pow = 10**(rand.uniform(SNR[0], SNR[1])/10)
    
    # N = [n1 n2 .. nL]
    N = rand.randn(*shape)*np.sqrt(0.5/db2pow)
    
    return Y + N

def generate_bulk_data(data_points, N, K, L, freq):
    data = np.zeros((data_points,L,2*N))
    labels = np.zeros((data_points,L,K))

    for i in range(data_points):
        l,d = generate_single_data(N, K, L, freq)
        data[i,:L,:N] = d.T.real
        data[i,:L,N:2*N] = d.T.imag
        labels[i,:L,:K] = l

    data = data.reshape(data_points*L, 2*N)
    labels = labels.reshape(data_points*L, K)
    
    return labels, data


def save_generated_data(filename, labels, data):
    
    
    with open(filename + '_data.npy', 'wb') as f:
        np.save(f, data)
        
    with open(filename + '_labels.npy', 'wb') as f:
        np.save(f, labels)


def load_generated_data(filename):
    with open(filename + '_data.npy', 'rb') as f:
        data = np.load(f)
        
    with open(filename + '_labels.npy', 'rb') as f:
        labels = np.load(f)
        
    return labels, data

def check_data_exists(filename):
    return isfile(filename + '_data.npy') and isfile(filename + '_labels.npy')

def normalize(labels, data):
    labels = labels/np.pi # normalize labels to [0,1]
    data = data/np.max(data)
    
    return labels, data