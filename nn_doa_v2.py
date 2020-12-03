import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import sqrt,pi
import numpy as np
from resnet import Residual

import data_generation_v2 as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]

training_size = 100000
validation_size = 0.1 # 10% of training size

def data_initialization(training_size, N, K, L, freq, theta_dist = 'uniform', sort=True):
    training = f"data/training_v2_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq, theta_dist, sort)
        dg.save_generated_data(training, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)
    
    training_labels, training_data = dg.generate_bulk_data(training_size, N, K, L, freq, theta_dist, sort)

    return training_labels, training_data

def normalize_add_wgn(data, snr):
    data = dg.apply_wgn(data, snr)
    
    return data


# antennas
N = 8

# users
K = [6]

# frequency
freq = 1e9

# bits
L = 500

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

learning_rate = 0.1
momentum = 0.9
        
output_size = 180
block_depth = 7

# train neural network based on the covariance of the Y data
def train_model_v2(N, K, L, freq, snr):
    training_labels, training_data = data_initialization(training_size, N, K, L, freq, sort=True)

    training_data = normalize_add_wgn(training_data, snr)
    
    C = np.matmul(training_data, np.transpose(training_data, axes=[0,2,1]))
        
    C = (C/np.linalg.norm(C)).reshape(len(C),N*N)
    
    training_data = np.zeros((len(C), 2*N*N))
    
    training_data[:len(C), :N*N] = np.real(C)
    training_data[:len(C), N*N:] = np.imag(C)
    
    training_data = training_data.reshape(len(training_data),2*N*N)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=False)
    print(training_labels.shape)

    # define model
    
    input_ = tf.keras.layers.Input(shape=[2*N*N])
    
    h1 = tf.keras.layers.Dense(output_size)(input_)
    h1 = tf.keras.layers.Reshape((output_size,1))(h1)
    h1 = Residual(block_depth, output_size)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)
    
    h1 = tf.keras.layers.Dense(2*output_size)(h1)
    h1 = tf.keras.layers.Reshape((2*output_size,1))(h1)
    h1 = Residual(block_depth, 2*output_size)(h1)
    h1 = tf.keras.layers.Dropout(0.5)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)
    
    h1 = tf.keras.layers.Dense(4*output_size)(h1)
    h1 = tf.keras.layers.Reshape((4*output_size,1))(h1)
    h1 = Residual(block_depth, 4*output_size)(h1)
    h1 = tf.keras.layers.Dropout(0.5)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)

    output = tf.keras.layers.Dense(output_size, activation='sigmoid')(h1)
    
    model = keras.Model(inputs=[input_], outputs=[output] )

    sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    #stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-6)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
    
    model.fit(training_data, training_labels, batch_size=800, epochs=100, validation_data=(validation_data, validation_labels))
    
    #tf.keras.utils.plot_model(
    #    model,
    #    to_file="model.png",
    #)

    #model.save(f"models/dnn_v2_users_{K}_bits_{L}_sgd_lr_{learning_rate}_momentum_{momentum}")

    #plt.xlabel('Epoch')
    #plt.ylabel('MSE')
    #plt.plot(m.history['loss'], 'r')
    #plt.plot(m.history['val_mean_squared_error'], 'b')
    #plt.legend(['Training', 'Validation'])
    #plt.savefig(f"figures/learning_curve_users_{K}_bits_{L}_sgd_lr_{learning_rate}_momentum_{momentum}.svg", format='svg')
    #plt.clf()
    
for k in K:
    train_model_v2(N, k, L, freq, snr)