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
    testing_labels, testing_data = dg.generate_bulk_data(testing_size, N, K, L, freq, dist, sort)

    return testing_labels, testing_data

def apply_wgn(Y):
    shape = Y.shape
    db2pow = 10**(30/10)
    
    # N = [n1 n2 .. nL]
    N = np.random.randn(*shape)*np.sqrt(0.5/db2pow)

    return Y+N

def normalize(labels, data):
    data = apply_wgn(data)
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


tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)


for dist in ['uniform','zeros','ones']:
    testing_labels, testing_data = data_initialization(N, K, L, freq, dist, True)
    testing_labels, testing_data = normalize(testing_labels, testing_data)
    
    #loss, mse = model.evaluate(testing_data, testing_labels, verbose=2)
    
    #print(mse)
    
    m = model.predict(testing_data)
    k = m*np.pi - pi/2
    testing_labels = testing_labels*np.pi - pi/2
    
    p = pos[dist]
    r = p[0]
    c = p[1]
    
    ax[r,c].set_title(dist)
    ax[r,c].set_xlabel('Theta_n')
    ax[r,c].set_ylabel('Variance')
    
    ax2[r,c].set_title(dist)
    ax2[r,c].set_xlabel('Theta_n')
    ax2[r,c].set_ylabel('Mean')
    
    for i in range(K):
        #ax[0,0].set_ylim(-0.2, 0.2)
        ax[r,c].plot((i+1)*np.ones_like(k[:,i].var()), k[:,i].var(), 'x', color='b')
        ax[r,c].plot((i+1)*np.ones_like(testing_labels[:,i].var()), testing_labels[:,i].var(), 'o', color='r')
        ax2[r,c].plot((i+1)*np.ones_like(k[:,i].mean()), k[:,i].mean(), 'x', color='b')
        ax2[r,c].plot((i+1)*np.ones_like(testing_labels[:,i].mean()), testing_labels[:,i].mean(), 'o', color='r')
    
    ax3[r,c].set_title(dist)
    ax3[r,c].set_xlabel('Mean difference of Theta_n and Theta_{n-1}')
    ax3[r,c].set_ylim(-0.2, 0.2)
    ax3[r,c].plot(np.diff(k).mean(axis=0), np.zeros_like(np.diff(k).mean(axis=0)) + 0.1, 'x', color='b')
    ax3[r,c].plot(np.diff(testing_labels).mean(axis=0), np.zeros_like(np.diff(testing_labels).mean(axis=0)) - 0.1, 'o', color='r')

fig.show()
fig2.show()
fig3.show()