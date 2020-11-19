import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import sqrt,pi
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

import data_generation as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]


testing_size = 1000

def data_initialization(N, K, L, freq, dist, sort):
    labels, data = dg.generate_bulk_data(testing_size, N, K, L, freq, dist, sort)

    return labels, data

def apply_wgn(Y, snr):
    shape = Y.shape
    db2pow = 10**(np.random.uniform(snr[0], snr[1])/10)
    
    # N = [n1 n2 .. nL]
    N = np.random.randn(*shape)*np.sqrt(0.5/db2pow)

    return Y+N

def normalize(labels, data, snr):
    data = apply_wgn(data, snr)
    labels = (labels - dg.min_theta)/(dg.max_theta - dg.min_theta) # normalize labels to [0,1]
    data = data/np.max(data)
    
    return labels, data

# antennas
N = 128

# users
K = 32#[4,8,16,32]

# frequency
freq = 1e9

# bits
L = 16

snr = [5, 30]

learning_rate = 0.1
momentum = 0.9

model = load_model(f"models/users_{K}_bits_{L}_sgd_lr_{learning_rate}_momentum_{momentum}")

pos = {
       'uniform': (0,0),
       'normal': (0,1),
       'zeros': (1,0),
       'ones': (1,1)
       }


fig, ax = plt.subplots(2,2,constrained_layout=True)
fig.set_size_inches(18, 10)

fig2, ax2 = plt.subplots(2,2,constrained_layout=True)
fig2.set_size_inches(18, 10)

fig3, ax3 = plt.subplots(2,2,constrained_layout=True)
fig3.set_size_inches(18, 10)

for dist in ['uniform','zeros','ones']:
    labels, data = data_initialization(N, K, L, freq, dist, True)
    labels, data = normalize(labels, data, snr)
    
    #loss, mse = model.evaluate(data, labels, verbose=2)
    
    #print(mse)
    
    m = model.predict(data)
    k = m*np.pi - pi/2
    labels = labels*np.pi - pi/2
    
    p = pos[dist]
    r = p[0]
    c = p[1]
    
    ax[r,c].set_title(dist)
    ax[r,c].set_xlabel('Theta_n')
    ax[r,c].set_ylabel('Variance')
    
    ax2[r,c].set_title(dist)
    ax2[r,c].set_xlabel('Theta_n')
    ax2[r,c].set_ylabel('Mean')
    
    ax3[r,c].set_title(dist)
    ax3[r,c].set_xlabel('Theta_n')
    ax3[r,c].set_ylabel('Mean abs delta theta')
    ax3[r,c].set_ylim([-0.01, 0.16])
    
    # plot mean and variance
    for i in range(K):
        ax[r,c].plot(i+1, k[:,i].var(), 'x', color='b')
        ax[r,c].plot(i+1, labels[:,i].var(), 'o', color='r')
        ax2[r,c].plot(i+1, k[:,i].mean(), 'x', color='b')
        ax2[r,c].plot(i+1, labels[:,i].mean(), 'o', color='r')
    
    # plot mean absolute difference between consecutive theta bins
    for i in range(1, K):
        diff_x = np.abs(k[:,i-1] - k[:,i]).mean()
        diff_o = np.abs(labels[:,i-1] - labels[:,i]).mean()
        ax3[r,c].plot(i+1, diff_x, 'x', color='b')
        ax3[r,c].plot(i+1, diff_o, 'o', color='r')
    
fig.show()
fig2.show()
fig3.show()