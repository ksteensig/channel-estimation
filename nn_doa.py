import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import sqrt,pi

import data_generation as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]

training_size = 150000
validation_size = 0.1 # 10% of training size

def data_initialization(training_size, N, K, L, freq):
    training = f"data/training_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq)
        dg.save_generated_data(training, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)

    return training_labels, training_data

def normalize_add_wgn(labels, data, snr):
    data = dg.apply_wgn(data, snr)

    labels, data = dg.normalize(labels, data)

    return labels, data


denormalize = lambda x: (sqrt(x)*pi)**2

# antennas
N = 128

# users
K = [8, 32]

# frequency
freq = 1e9

# bits
L = 16

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

learning_rate = 0.1
momentum = 0.5

def train_model(N, K, L, freq, snr):
    training_labels, training_data = data_initialization(training_size, N, K, L, freq)

    training_labels, training_data = normalize_add_wgn(training_labels, training_data, snr)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size, shuffle=False)

# define model
    model = keras.Sequential([
            keras.layers.Dense(2*N, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(300, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.1, input_shape=(200,)),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(K, activation='sigmoid')
            ])

    sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-6)

    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    m = model.fit(training_data, training_labels, batch_size=1200, epochs=1500, validation_data=(validation_data, validation_labels), callbacks=[stopping])

    model.save(f"models/users_{K}_sgd_lr_{learning_rate}_momentum_{momentum}")

    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(denormalize(m.history['mean_squared_error']), 'r')
    plt.plot(denormalize(m.history['val_mean_squared_error']), 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig(f"figures/learning_curve_users_{K}_sgd_lr_{learning_rate}_momentum_{momentum}.svg", format='svg')
    plt.clf()
    

for k in K:
    train_model(N, k, L, freq, snr)