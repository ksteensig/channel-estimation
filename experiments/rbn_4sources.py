import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from resnet import ResnetBlock

import rbn_datagen as dg

print('TensorFlow Version:', tf.__version__)


def data_initialization(training_size, N, K, L, freq, theta_dist = 'uniform', sort=False):
    training = f"data/RBN_training_N={N}_K={K}_L={L}_sort={sort}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq, theta_dist, sort)
        dg.save_generated_data(training, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)

    return training_labels, training_data

def normalize_add_wgn(labels, data, snr):
    data = dg.apply_wgn(data, 16, snr)

    dg.normalize(labels, data)


training_size = 50000
validation_size = 0.1 # 10% of training size

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
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate/(2**np.floor(np.log10(epoch)))

def train_model(N, K, L, freq, snr, sort):
    training_labels, training_data = data_initialization(training_size, N, K, L, freq, sort = sort)

    normalize_add_wgn(training_labels, training_data, snr)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=False)

    # define model
    model = keras.Sequential([
            keras.layers.Dense(2*K),
            keras.layers.Dense(2*K),
            keras.layers.Dense(K, activation='sigmoid')
            ])

    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-5)

    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    m = model.fit(training_data, training_labels, batch_size=1200, epochs=300, validation_data=(validation_data, validation_labels), callbacks=[stopping])

    model.save(f"models/RBN_N={N}_K={K}_L={L}_lr={learning_rate}")

    plt.xlabel('Epoch')
    plt.ylabel('Normalized MSE')
    plt.plot(m.history['mean_squared_error'], 'r')
    plt.plot(m.history['val_mean_squared_error'], 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig(f"figures/RBN_learning_curve_N={N}_K={K}_L={L}_lr={learning_rate}.svg", format='svg')
    plt.clf()
    
    return model
    

model = train_model(N, K, L, freq, snr, sort = True)
    