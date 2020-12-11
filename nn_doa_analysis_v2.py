import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import sqrt,pi
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

import data_generation_v2 as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]

def apply_wgn(Y, SNR):
    shape = Y.get_shape()
    db2pow = 10**(np.random.uniform(SNR[0], SNR[1])/10)
    
    # N = [n1 n2 .. nL]
    N1 = tf.random.normal(shape)*np.sqrt(0.5/db2pow)
    N2 = tf.random.normal(shape)*np.sqrt(0.5/db2pow)
    
    return Y + tf.complex(N1, N2)

def normalize_add_wgn(data, snr):
    data = apply_wgn(data, snr)
    
    return data


testing_size = 20

def data_initialization(training_size, N, K, L, freq, theta_dist = 'uniform'):
    
    labels, data = dg.generate_bulk_data(training_size, N, K, L, freq, theta_dist)
        
    training_data = tf.convert_to_tensor(data)
    training_labels = tf.convert_to_tensor(labels)

    training_data = normalize_add_wgn(training_data, snr)
        
    C = tf.matmul(training_data, tf.transpose(training_data, perm=[0,2,1]))
        
    C = tf.reshape(C, [len(C),N*N])
        
    C = C/tf.reshape(tf.norm(C,axis=1), (-1, 1))
    
    r,i = tf.math.real(C), tf.math.imag(C)
    
    training_data = tf.cast(tf.concat([r, i], axis=1), dtype=tf.float32)
    
    return training_labels, training_data

# antennas
N = 8

# users
K = 6

# frequency
freq = 1e9

# bits
L = 500

snr = [15, 15]

learning_rate = 0.1

labels, data = data_initialization(testing_size, N, K, L, freq)

model = load_model(f"models/dnn_v2_users_{K}_bits_{L}_sgd_lr_{learning_rate}", custom_objects={ 'loss_fun': loss_fun })

print(model.evaluate(data, labels))

predictions = tf.convert_to_tensor(model.predict(data))
plt.plot(np.mean(predictions, axis=0))

l = tf.cast(labels, tf.float32)

l = tf.map_fn(loss_fun_body, l)
plt.plot(np.mean(l, axis=0))

#h = pd.read_csv('history.csv')

#plt.plot(h['loss'])
#plt.plot(h['val_loss'])