import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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
N = 128

# users
K = 8

# frequency
f = 1e9

training = []
labels = []

for i in range(40000):
    label,_,train = generate_los_ula_data(N, K, 1, 10, f)
    training.append(train.T)
    labels.append(label.T)
    
training = np.array(training)
labels = np.array(labels)

training = training/abs(np.max(training))

# define model
model = keras.Sequential([
    keras.layers.Dense(N, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1, input_shape=(200,)),
    keras.layers.GaussianNoise(1),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(K, activation='linear')
])
    

# using Adam doesn't require normalizing the data
opt = keras.optimizers.SGD(learning_rate=0.024, decay=0.96)

model.compile(optimizer=opt,
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])


model.fit(training, labels, batch_size=1200, epochs=20)

"""
testing = []
labels = []

for i in range(10000):
    label,_,test = generate_los_ula_data(N, K, 1, 10, f)
    testing.append(np.abs(test.T))
    labels.append(label.T)
    
testing = np.array(testing)
labels = np.array(labels)

testing = testing/np.max(testing)

test_loss, mse = model.evaluate(testing,  labels, verbose=2)

print('\n MSE:', mse)
"""
