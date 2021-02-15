import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from resnet import ResnetBlock

import cbn_datagen as dg

print('TensorFlow Version:', tf.__version__)

training_size = 500000
validation_size = 0.1 # 10% of training size

resolution = 180

# receiver antennas
N = 16

# users
K = 4

# frequency
freq = 1e9

# bits
L = 16

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

learning_rate = 0.001

def adaptive_learning_rate(epoch):
    return learning_rate/(2**np.floor(epoch/10))

def train_model(N, K, L, freq, snr, resolution):
    training_labels, training_data = dg.data_initialization(training_size, N, K, L, freq, resolution, snr, cache=True)
    
    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=False)
    
    # define model
    
    model = keras.Sequential([
            keras.layers.Dense(2*N, activation='relu'),
            keras.layers.Dense(2*N, activation='relu'),    
            keras.layers.Dense(2*N, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(resolution, activation='sigmoid', activity_regularizer=keras.regularizers.l1(l1=0.01))            ])
    """
    model = keras.Sequential([
            keras.layers.Dense(resolution),
            ResnetBlock(resolution, 7),
            keras.layers.Dense(2*K),
            keras.layers.Dense(resolution, activation='sigmoid', activity_regularizer=keras.regularizers.l1(l1=0.01))
            ])
    """
    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4)
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)

    model.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy())

    m = model.fit(training_data, training_labels, batch_size=32, epochs=300, validation_data=(validation_data, validation_labels), callbacks=[stopping, lrate])

    model.save(f"models/CBN_N={N}_K={K}_L={L}_lr={learning_rate}")
    """
    plt.xlabel('Epoch')
    plt.ylabel('Normalized MSE')
    plt.plot(m.history['mean_squared_error'], 'r')
    plt.plot(m.history['val_mean_squared_error'], 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig(f"figures/RBN_learning_curve_N={N}_K={K}_L={L}_lr={learning_rate}.svg", format='svg')
    plt.clf()
    """
    return model
    

model = train_model(N, K, L, freq, snr, resolution)