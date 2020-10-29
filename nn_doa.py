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

training_size = 100000
validation_size = 10000
testing_size = 20000

def data_initialization(N, K, L, freq, snr):
    training = f"data/training_{N}_{K}_{L}"
    #testing = f"data/testing_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq)
        dg.save_generated_data(training, labels, data)
    #if not dg.check_data_exists(testing):
    #    labels, data = dg.generate_bulk_data(testing_size, N, K, L, freq)
    #    dg.save_generated_data(testing, labels, data)
        
    training_labels, training_data  = dg.load_generated_data(training)
    #testing_labels, testing_data = dg.load_generated_data(testing)
    
    training_data = dg.apply_wgn(training_data, snr)
    #testing_data = dg.apply_wgn(testing_data, snr)

    training_labels, training_data = dg.normalize(training_labels, training_data)
    #testing_labels, testing_data = dg.normalize(testing_labels, testing_data)

    return training_labels, training_data

# antennas
N = 128

# users
K = 4

# frequency
freq = 1e9

# bits
L = 16

# snr between 5 and 30
snr = [5, 30]

training_labels, training_data = data_initialization(N, K, L, freq, snr)

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
sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=0.5)
adam = keras.optimizers.Adam(learning_rate=0.01)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-6)

model.compile(optimizer=sgd,
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

m = model.fit(training_data, training_labels, batch_size=1200, epochs=2500, validation_data=(validation_data, validation_labels))

model.save('models/users_4_snr_5_30_sgd_lr_0.1_momentum_0.5')

plt.plot((sqrt(model.history.history['mean_squared_error'])*pi)**2, 'r')
plt.plot((sqrt(model.history.history['val_mean_squared_error'])*pi)**2, 'b')
plt.legend(['Training', 'Validation'])
plt.show()

#test_loss, mse = model.evaluate(testing_data, testing_labels, verbose=2)

#print('\n MSE:', (sqrt(mse)*pi)**2)
