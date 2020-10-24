import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import data_generation as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]

training_size = 10000
validation_size = 1000
testing_size = 2000

def data_initialization(N, K, L, freq, snr):
    training = f"data/training_{N}_{K}_{L}"
    testing = f"data/testing_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq)
        dg.save_generated_data(training, labels, data)
    if not dg.check_data_exists(testing):
        labels, data = dg.generate_bulk_data(testing_size, N, K, L, freq)
        dg.save_generated_data(testing, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)
    testing_labels, testing_data = dg.load_generated_data(testing)
    
    training_data = dg.apply_wgn(training_data, snr)
    testing_data = dg.apply_wgn(testing_data, snr)

    training_labels, training_data = dg.normalize(training_labels, training_data)
    testing_labels, testing_data = dg.normalize(testing_labels, testing_data)

    return training_labels, training_data, testing_labels, testing_data

# antennas
N = 128

# users
K = 8

# frequency
freq = 1e9

# bits
L = 16

# snr between 5 and 30
snr = [100, 100]

training_labels, training_data, testing_labels, testing_data = data_initialization(N, K, L, freq, snr)

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

# using Adam doesn't require normalizing the data
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.1)
adam = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=adam,
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

m = model.fit(training_data, training_labels, batch_size=1200, epochs=30, validation_data=(validation_data, validation_labels))

plt.plot(m.history['mean_squared_error'], 'r')
plt.plot(m.history['val_mean_squared_error'], 'b')
plt.legend(['Training', 'Validation'])
plt.show()


test_loss, mse = model.evaluate(testing_data, testing_labels, verbose=2)

print('\n MSE:', mse)
