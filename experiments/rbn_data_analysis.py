import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import sqrt,pi
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model
import numpy as np

plt.rcParams['text.usetex'] = True

import rbn_datagen as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]


testing_size = 1000

# antennas
N = 16

# users
K = 8

# frequency
freq = 2.4e9

# bits
L = 16

snr = [5, 30]

model = load_model(f"models/RBN_N={N}_K={K}_L={L}")

pos = {
       'uniform': (0,0),
       'zeros': (1,0),
       'ones': (2,0)
       }


matplotlib.rc('font', size=25)

plt.rc('text', usetex=True)
fig, ax = plt.subplots(3,1,constrained_layout=True)
fig.set_size_inches(18, 18)

fig2, ax2 = plt.subplots(3,1,constrained_layout=True)
fig2.set_size_inches(18, 18)

for dist in ['uniform','zeros','ones']:
    labels, data = dg.generate_bulk_data(testing_size, N, K, L, freq, dist, sort = True)
    data = dg.apply_wgn(data, L, snr)
    dg.normalize(labels, data)
    
    #loss, mse = model.evaluate(data, labels, verbose=2)
    
    #print(mse)
    
    m = model.predict(data)
    k = m*np.pi - pi/2
    labels = labels*np.pi - pi/2
    
    p = pos[dist]
    r = p[0]
    
    ax2[r].set_ylim([-np.pi/2-0.01, np.pi/2+0.01])
    
    ax[r].set_title(dist)
    ax[r].set_xlabel(r'$\theta_k$')
    ax[r].set_ylabel(r'$\mathrm{var}(error_k)$')
    
    ax2[r].set_title(dist)
    ax2[r].set_xlabel(r'$\theta_k$')
    ax2[r].set_ylabel(r'$\mathrm{mean}(\theta_k)$')
    
    # plot mean and variance
    for i in range(K):
        xlabel = r'$\theta_{' + str(i+1) + '}$'
        ax[r].plot(xlabel, (k[:,i]).var(), 'x', color='b')
        ax[r].plot(xlabel, labels[:,i].var(), 'o', color='r')
        ax2[r].plot(xlabel, k[:,i].mean(), 'x', color='b')
        ax2[r].plot(xlabel, labels[:,i].mean(), 'o', color='r')
    

fig.savefig('nn_doa_v1_variance.eps', format='eps')
fig2.savefig('nn_doa_v1_mean.eps', format='eps')    

fig.show()
fig2.show()