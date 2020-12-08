import numpy as np
from os.path import isfile

max_theta = np.deg2rad(75)
min_theta = -max_theta

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

# SNR is a range between min and max SNR in dB
def generate_single_data(N, K, L, f, theta_dist = 'uniform', resolution=180):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl
    
    # steering vector
    array_response = lambda array,theta: np.exp(-1j*2*np.pi*array*np.sin(theta))*np.sqrt(1/N)

    theta = np.zeros((K,1))
    
    if theta_dist == 'uniform':
        theta = (2*max_theta)*np.random.rand(K,1) + min_theta
    elif theta_dist == 'normal':
        theta = np.random.randn(K,1)
    elif theta_dist == 'zeros':
        pass
    elif theta_dist == 'ones':
        theta = np.ones((K,1))
    
    # realizations of received data, with L being number of realizations
    # Y = [y1 y2 .. yL]
    Y = np.zeros((N,L)) + 1j*np.zeros((N,L))
    
    # random inputs with |x_{k,l}| = 1
    for l in range(L):
        for k in range(K):
            alpha = (np.random.randn(1) + 1j*np.random.randn(1))*np.sqrt(1/2)
            response = array_response(array, theta[k])
            Y[:,l] += alpha*response
            
    theta = np.floor(((theta - min_theta)/(max_theta - min_theta)*180)).astype(int)
    
    theta_grid = np.zeros((resolution, 1))
    theta_grid[theta] = 1
        
    return theta_grid, Y

def generate_bulk_data(data_points, N, K, L, freq, dist='uniform', resolution=180):
    data = np.zeros((data_points,N,L), dtype='complex64')
    labels = np.zeros((data_points,resolution,1))

    for i in range(data_points):
        l,d = generate_single_data(N, K, L, freq, dist, resolution)
        data[i,:N,:L] = d
        labels[i,:resolution] = l
        
    labels = labels.reshape(data_points,resolution)
            
    return labels, data