import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

import cbn_datagen as dg

def train_model(C = 32, N = 16, K = 8, L = 16, freq = 2.4e9, training_size = 200000, validation_size = 0.1, learning_rate = 0.001, snr = [5, 30]):
    
    training_labels, training_data = dg.data_initialization(training_size, N, K, L, freq, 180, snr, cache=True)
    
    training_size = int(len(training_data)/L)
    
    training_data = dg.apply_wgn(training_data, L, snr)
    training_data = training_data.reshape((training_size, N*L))
    training_data = np.concatenate((training_data.real,training_data.imag), axis=1)
    #training_data = training_data / np.max(np.abs(training_data), axis=1).reshape((training_size,1))
    training_data = training_data - np.min(training_data, axis=1).reshape((training_size,1))
    training_data = training_data / np.max(np.abs(training_data), axis=1).reshape((training_size,1))
    
    print(training_data.shape)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=False)

    training_labels = None
    validation_labels = None

    # define model
    encoder = keras.Sequential([
            keras.layers.Dense(2*N*L, 'relu'),
            keras.layers.Dense(2*N*L, 'relu'),
            keras.layers.Dense(2*N, 'relu'),
            keras.layers.Dense(C)
            ])
    
    decoder = keras.Sequential([
            keras.layers.Dense(2*N, 'relu'),
            keras.layers.Dense(2*N*L, 'relu'),
            keras.layers.Dense(2*N*L, 'sigmoid'),
            ])
    
    
    auto_input = tf.keras.Input(shape=(2*N*L))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto_encoder = tf.keras.Model(auto_input, decoded)
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))

    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4)

    auto_encoder.compile(optimizer=adam,
                  loss='mse')

    m = auto_encoder.fit(training_data, training_data, batch_size=128, epochs=100, validation_data=(validation_data, validation_data), callbacks=[stopping, lrate])

    with open(f"history/AE_C={C}_N={N}_K={K}_L={L}", 'wb') as f:
        pickle.dump(m.history, f)


    auto_encoder.save(f"models/AE_C={C}_N={N}_K={K}_L={L}")    
    
    
    return auto_encoder
    
