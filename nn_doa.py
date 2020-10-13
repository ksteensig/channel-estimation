# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np

from data_gen import generate_los_ula_data

print(tf.__version__)


# load data
#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



# normalize data
#train_images = train_images / 255.0
#test_images = test_images / 255.0

# antennas
N = 8

# users
K = 4

# frequency
f = 1e9

# retain probability
p = 0.9

# define model
model = keras.Sequential([
    keras.layers.Dense(N, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(p),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(K, activation='linear')
])

training = []
labels = []

for i in range(100000):
    label,_,train = generate_los_ula_data(N, K, 1, 20, f)
    training.append(train.T)
    labels.append(label.T)
    
training = np.array(training)
labels = np.array(labels)


model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])


model.fit(training, labels, epochs=10)


testing = []
labels = []

for i in range(10000):
    label,_,test = generate_los_ula_data(N, K, 1, 20, f)
    testing.append(test.T)
    labels.append(label.T)
    
testing = np.array(testing)
labels = np.array(labels)

test_loss, mse = model.evaluate(testing,  labels, verbose=2)

print('\n MSE:', mse)