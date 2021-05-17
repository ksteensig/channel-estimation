import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from tensorflow.keras.models import load_model

import cbn_recv_datagen as dg

# N: receiver antennas
# K: users
# L: bits
# freq: frequency
# snr between 5 and 30
def train_model(N, K, L, freq = 2.4e9, snr = [5, 30], resolution = 180, training_size = 200000, validation_size = 0.1, learning_rate = 0.001):
    #training_labels, training_data, validation_labels, validation_data = dg.data_initialization(training_size, N, K, L, freq, resolution, cache=False)
    
    tsize = int((1-validation_size)*training_size)
    vsize = int((validation_size)*training_size)
    
    training_labels, training_data = dg.generate_bulk_data(tsize, N, K, L)
    validation_labels, validation_data = dg.generate_bulk_data(vsize, N, K, L)
    print(training_data.shape)
    
    training_data = np.concatenate((training_data.real,training_data.imag), axis=1)
    training_data = dg.apply_wgn(training_data, L, snr)
    training_data = training_data.reshape((tsize, 2*N*L))
    training_data = training_data / np.max(np.abs(training_data), axis=1).reshape((tsize,1))
    
    validation_data = np.concatenate((validation_data.real,validation_data.imag), axis=1)
    validation_data = dg.apply_wgn(validation_data, L, snr)
    validation_data = validation_data.reshape((vsize, 2*N*L))
    validation_data = validation_data / np.max(np.abs(validation_data), axis=1).reshape((vsize,1))
    
    #ae = load_model(f"models/AE_N={N}_K={K}_L={L}")
    
    #encoded = ae.get_layer('sequential_14')
                
    # define model
    model = keras.Sequential([
            #encoded,
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
    
    m = model.fit(training_data, training_labels, batch_size=32, epochs=300, validation_data=(validation_data, validation_labels), callbacks=[stopping, lrate])

    with open(f"history/CBN_recv_full_N={N}_K={K}_L={L}", 'wb') as f:
        pickle.dump(m.history, f)

    model.save(f"models/CBN_recv_full_N={N}_K={K}_L={L}")

    return model