import numpy as np
import numpy.random as rand
from os.path import isfile

# SNR is a range between min and max SNR in dB
def generate_single_data(N, K, f, theta_bound = np.pi/2, theta_dist = 'uniform', sort = False):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
        
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl
    
    # steering vector
    array_response = lambda array,theta: np.exp(-1j*2*np.pi*array*np.sin(theta))*np.sqrt(1/N)

    theta = np.zeros((K,1))
    
    if theta_dist == 'uniform':
        theta = 2*theta_bound*np.random.rand(K,1) - theta_bound
    elif theta_dist == 'normal':
        theta = np.random.randn(K,1)
    elif theta_dist == 'zeros':
        pass
    elif theta_dist == 'ones':
        theta = np.ones((K,1))
    
    
    alpha = (np.random.randn(K,1) + 1j*np.random.randn(K,1))*np.sqrt(1/2)
    response = array_response(array, theta)
        
    yl = np.inner(response.T, alpha.T).T
    
    yl_split = np.concatenate((yl.real,yl.imag), axis=1)
    
    if sort:
        theta = np.sort(theta, axis=0)
        
    return theta.T, yl_split

def apply_wgn(Y, L, SNR):
    shape = Y.shape

    # Y consists of L-repeats of different y's
    # all identical y's must use the same SNR
    db2pow = 10**(rand.uniform(SNR[0], SNR[1], size=(int(shape[0]/L),1))/10)
    db2pow = np.repeat(db2pow, L, axis=1)
    db2pow = db2pow.flatten().reshape((shape[0], 1))
        
    return Y + rand.randn(*shape)*np.sqrt(0.5/db2pow)

def generate_bulk_data(data_points, N, K, L, freq, dist = 'uniform', sort = False):
    data = np.zeros((data_points*L,2*N))
    labels = np.zeros((data_points*L,K))
    
    for i in range(data_points):
        theta,yl = generate_single_data(N, K, freq, theta_bound=np.pi/2, theta_dist = dist,sort = sort)
        Theta, Y = np.repeat(theta, L, axis=0), np.repeat(yl, L, axis=0)
        
        start = L*i
        end = start + L
        
        data[start:end, :] = Y
        labels[start:end, :] = Theta
            
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

def normalize(labels, data, theta_bound = np.pi/2):
    max_theta = theta_bound
    min_theta = -max_theta
        
    labels[:] = (labels[:] - min_theta)/(max_theta - min_theta) # normalize labels to [0,1]
    data[:] = data[:]/np.max(data[:])
    
    #return labels,data