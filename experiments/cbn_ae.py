import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from resnet import ResnetBlock
import pickle
from tensorflow.keras.models import load_model

import cbn_ae_datagen as dg

# N: receiver antennas
# K: users
# L: bits
# freq: frequency
# snr between 5 and 30
def train_model(C = 32, N = 16, K = 4, L = 16, freq = 2.4e9, snr = [5, 30], resolution = 180, training_size = 200000, validation_size = 0.1, learning_rate = 0.001):
    training_labels, training_data = dg.data_initialization(training_size, N, K, L, freq, resolution, snr, cache=True)    
    training_size = int(len(training_data)/L)
    
    training_data = dg.apply_wgn(training_data, L, snr)
    training_data = training_data.reshape((training_size, N*L))
    training_data = np.concatenate((training_data.real,training_data.imag), axis=1)
    training_data = training_data - np.min(training_data, axis=1).reshape((training_size,1))
    training_data = training_data / np.max(np.abs(training_data), axis=1).reshape((training_size,1))

    ae = load_model(f"models/AE_C={C}_N={N}_K={K}_L={L}")
    
    encoder = ae.layers[1] # get second layer which is the encoder
    
    training_data = encoder(training_data)
                
    # define model
    model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(resolution, activation='sigmoid')
            ])
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))
    
    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-5)
    lrate = keras.callbacks.LearningRateScheduler(adaptive_learning_rate)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy')
    
    m = model.fit(training_data, training_labels, batch_size=32, epochs=300, validation_split=validation_size, callbacks=[stopping, lrate])

    with open(f"history/CBN_ae_out_C={C}_N={N}_K={K}_L={L}", 'wb') as f:
        pickle.dump(m.history, f)

    model.save(f"models/CBN_ae_out_C={C}_N={N}_K={K}_L={L}")

    return model
