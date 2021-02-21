import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from resnet import ResnetBlock
import pickle

import cbn_recv_datagen as dg

# N: receiver antennas
# K: users
# L: bits
# freq: frequency
# snr between 5 and 30
def train_model(N, K, L, freq = 2.4e9, snr = [5, 30], resolution = 180, training_size = 500000, validation_size = 0.1, learning_rate = 0.001):
    training_labels, training_data = dg.data_initialization(training_size, N, K, L, freq, resolution, snr, cache=True)    
    
    dg.normalize_add_wgn(training_labels, training_data, L, snr)
    
    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=True)
    
    # define model
    model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(resolution, activation='sigmoid')
            ])
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))
    
    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4)
    lrate = keras.callbacks.LearningRateScheduler(adaptive_learning_rate)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    m = model.fit(training_data, training_labels, batch_size=32, epochs=300, validation_data=(validation_data, validation_labels), callbacks=[stopping, lrate])

    with open(f"history/CBN_recv_N={N}_K={K}_L={L}", 'wb') as f:
        pickle.dump(m.history, f)

    model.save(f"models/CBN_recv_N={N}_K={K}_L={L}")

    return model