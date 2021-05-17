import numpy as np
import numpy.random as rand
from os.path import isfile

# SNR is a range between min and max SNR in dB
def generate_single_data(N, K, f, res=180, theta_bound = np.pi/2, theta_ = None):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
        
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl
    
    # steering vector
    array_response = lambda array,theta: np.exp(-1j*2*np.pi*array*np.sin(theta))*np.sqrt(1/N)

    if theta_ is None:
        theta = 2*theta_bound*np.random.rand(K,1) - theta_bound
    else:
        theta = theta_.copy().reshape((K,1))
    
    
    alpha = (np.random.randn(K,1) + 1j*np.random.randn(K,1))*np.sqrt(1/2)
    response = array_response(array, theta)
        
    yl = np.inner(response.T, alpha.T).T
    
    #yl_split = np.concatenate((yl.real,yl.imag), axis=1)
    
    theta = np.floor(((theta - theta_bound)/(2*theta_bound)*res)).astype(int)
    
    theta_grid = np.zeros((res, 1))
    theta_grid[theta] = 1
        
    return theta_grid.T, yl

def apply_wgn(Y, L, SNR):
    shape = Y.shape

    # Y consists of L-repeats of different y's
    # all identical y's must use the same SNR
    db2pow = 10**(rand.uniform(SNR[0], SNR[1], size=(int(shape[0]/L),1))/10)
    
    db2pow = np.repeat(db2pow, L, axis=1)
    db2pow = db2pow.flatten().reshape((shape[0], 1))
    N = rand.randn(*shape)*np.sqrt(0.5/db2pow)
        
    return Y + N

def generate_bulk_data(data_points, N, K, L, freq = 2.4e9, res=180, theta_ = None):
    #data = np.zeros((data_points*L, 2*N))
    #labels = np.zeros((data_points*L, res))
    data = np.zeros((data_points*L, N), dtype=np.complex64)
    labels = np.zeros((data_points, res))
    
    for i in range(data_points):
        if theta_ is None:
            theta = None
        else:
            theta = theta_[:,i]
            
        theta, yl = generate_single_data(N, K, freq, res, theta_bound=np.pi/2, theta_ = theta)
        #Theta, Y = np.repeat(theta, L, axis=0), np.repeat(yl, L, axis=0)
        Y = np.repeat(yl, L, axis=0)
        start = L*i
        end = start + L
        
        data[start:end, :] = Y
        #labels[start:end, :] = Theta
        labels[i, :] = theta
            
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
    
def data_initialization(training_size, N, K, L, freq, res, snr, theta_dist = 'uniform', cache = True):
    training = f"data/CBN_recv_training_N={N}_K={K}_L={L}"
    validation = f"data/CBN_recv_validation_N={N}_K={K}_L={L}"
    
    if not cache:
        labels, data = generate_bulk_data(training_size, N, K, L, freq, res, theta_dist)
        #v_labels, v_data = generate_bulk_data(int(0.1 * training_size), N, K, L, freq, res, theta_dist)
        return labels, data#, v_labels, v_data
    
    if not check_data_exists(training):
        labels, data = generate_bulk_data(int(0.9 * training_size), N, K, L, freq, res, theta_dist)
        v_labels, v_data = generate_bulk_data(int(0.1 * training_size), N, K, L, freq, res, theta_dist)
        save_generated_data(training, labels, data)
        save_generated_data(validation, v_labels, v_data)
        return labels, data, v_labels, v_data
        
    training_labels, training_data  = load_generated_data(training)
    validation_labels, validation_data = load_generated_data(validation)

    return training_labels, training_data, validation_labels, validation_data

def normalize_add_wgn(data, L, snr):
    data = apply_wgn(data, L, snr)
    data = data/np.max(data, axis=1).reshape(len(data), 1)
    
    return data