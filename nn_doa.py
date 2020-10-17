import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_gen import generate_los_ula_data

print('TensorFlow Version:', tf.__version__)

# antennas
N = 128

# users
K = 8

# frequency
f = 1e9

# bits
L = 16

# snr between 5 and 30
snr = [5, 30]

training_size = 50000
validation_size = 10000

data = np.zeros((training_size,L,N), dtype='complex128')
labels = np.zeros((training_size,L,K), dtype='complex128')

for i in range(training_size):
    l,d = generate_los_ula_data(N, K, L, snr, f)
    data[i,:L,:N] = d.T
    labels[i,:L,:K] = l

data = data.reshape(training_size*L, N)
labels = labels.reshape(training_size*L, K)

data = data/np.max(abs(data))
labels = labels/np.pi # normalize labels to [0,1]

test_size=validation_size/training_size

t_data, v_data, t_labels, v_labels = train_test_split(data, labels, test_size=test_size, shuffle=False)

# define model
model = keras.Sequential([
    keras.layers.Dense(N, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1, input_shape=(200,)),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(K, activation='sigmoid')
])
    

# using Adam doesn't require normalizing the data
opt = keras.optimizers.SGD(learning_rate=0.024, decay=0.96)

model.compile(optimizer=opt,
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

batch_size = int(((training_size-validation_size)*L)/1200)

epochs = 50

t_mse = np.zeros((epochs,1))
v_mse = np.zeros((epochs,1))

for i in range(epochs):
    m = model.fit(t_data, t_labels, batch_size=batch_size, epochs=1, validation_data=(v_data, v_labels))
    t_mse[i] = m.history['mean_squared_error'][0]
    v_mse[i] = m.history['val_mean_squared_error'][0]
    

"""
testing = []
labels = []

for i in range(10000):
    label,test = generate_los_ula_data(N, K, L, 10, f)
    testing.append(np.abs(test.T))
    labels.append(label.T)
    
testing = np.array(testing)
labels = np.array(labels)

testing = testing/abs(np.max(testing))
labels = labels/np.pi

test_loss, mse = model.evaluate(testing,  labels, verbose=2)

print('\n MSE:', mse)
"""
