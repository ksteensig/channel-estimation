import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle

import rbn_cov_datagen as dg

def train_model(N = 16, K = 4, L = 16, freq = 2.4e9, training_size = 500000, validation_size = 0.1, learning_rate = 0.001, snr = [5, 30], sort = False):    
    training_labels, training_data = dg.data_initialization(training_size, N, K, L, freq, snr)

    dg.normalize_add_wgn(training_labels, training_data, snr)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=True)

    # define model
    model = keras.Sequential([
            keras.layers.Dense(128, 'relu'),
            keras.layers.Dense(128, 'relu'),
            keras.layers.Dense(K, activation='sigmoid')
            ])
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))

    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    sgd = keras.optimizers.SGD(learning_rate=learning_rate)
    
    
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4)

    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    m = model.fit(training_data, training_labels, batch_size=128, epochs=300, validation_data=(validation_data, validation_labels), callbacks=[stopping, lrate])

    with open(f"history/RBN_cov_N={N}_K={K}_L={L}", 'wb') as f:
        pickle.dump(m.history, f)


    model.save(f"models/RBN_cov_N={N}_K={K}_L={L}")    
    

    return model
    