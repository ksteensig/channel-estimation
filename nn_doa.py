import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
import numpy as np

from sklearn.model_selection import train_test_split

from data_gen import generate_los_ula_data

print(tf.__version__)




# load data
#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



# normalize data
#train_images = train_images / 255.0
#test_images = test_images / 255.0

# antennas
N = 128

# users
K = 8

# frequency
f = 1e9

# bits
L = 16

# snr
snr = 10

training_size = 60000
validation_size = 20000

training = np.zeros((training_size,L,N), dtype='complex128')
labels = np.zeros((training_size,L,K), dtype='complex128')

for i in range(training_size):
    label,train = generate_los_ula_data(N, K, L, snr, f)
    training[i,:L,:N] = train.T
    labels[i,:L,:K] = label

training = training.reshape(training_size*L, N)
labels = labels.reshape(training_size*L, K)

training = training/np.max(abs(training))
labels = labels/np.pi


# define model
model = keras.Sequential([
    keras.layers.Dense(N, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.9, input_shape=(200,)),
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

model.fit(training, labels, batch_size=batch_size, epochs=10)

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
