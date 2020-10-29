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

training_size = 10000
validation_size = 1000
testing_size = 2000

def data_initialization(N, K, L, freq):
    training = f"data/training_{N}_{K}_{L}"
    #testing = f"data/testing_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq)
        dg.save_generated_data(training, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)

    return training_labels, training_data

def normalize_add_wgn(labels, data, snr):
    data = dg.apply_wgn(data, snr)

    labels, data = dg.normalize(labels, data)

    return labels, data

# antennas
N = 128

# users
K = 8

# frequency
freq = 1e9

# bits
L = 16

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

labels, data = data_initialization(N, K, L, freq)

training_labels, training_data = normalize_add_wgn(labels, data, snr)

training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size=validation_size/training_size, shuffle=False)

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

learning_rate = 0.1
momentum = 0.5
    
# using Adam doesn't require normalizing the data
sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
adam = keras.optimizers.Adam(learning_rate=0.01)

stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-5)

model.compile(optimizer=sgd,
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

m = model.fit(training_data, training_labels, batch_size=1200, epochs=2500, validation_data=(validation_data, validation_labels), callbacks=[stopping])

model.save(f"models/users_{K}_sgd_lr_{learning_rate}_momentum_{momentum}")

plt.plot((sqrt(model.history.history['mean_squared_error'])*pi)**2, 'r')
plt.plot((sqrt(model.history.history['val_mean_squared_error'])*pi)**2, 'b')
plt.legend(['Training', 'Validation'])
plt.show()

#test_loss, mse = model.evaluate(testing_data, testing_labels, verbose=2)

#print('\n MSE:', (sqrt(mse)*pi)**2)
