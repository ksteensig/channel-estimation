import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import sqrt,pi
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks

import cbn_ae_datagen as dg

print('TensorFlow Version:', tf.__version__)

# receiver antennas
N = 16

# users
K = 4

# frequency
freq = 2.4e9

# bits
L = 16

min_snr = 30
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

learning_rate = 0.001

resolution = 180

labels, data = dg.data_initialization(1, N, K, L, freq, 180, snr, cache=False)
    
training_size = int(len(data)/L)
    
#data = dg.apply_wgn(data, L, snr)

data2 = np.mean(data.copy(), axis=0)

data = data.reshape((training_size, N*L))

data = np.concatenate((data.real,data.imag), axis=1)
data = data / np.max(np.abs(data), axis=1).reshape((training_size,1))

model = load_model(f"models/CBN_ae_N={N}_K={K}_L={L}")


prediction = model.predict(data)

res = prediction[0] / np.max(np.abs(prediction[0]))

plt.figure(0)

plt.plot(labels[0], color='b')
plt.plot(res, color='r')
peaks = find_peaks(res, 0.05)[0]
plt.plot(peaks, res[peaks], 'x')



#plt.show()

#h = pd.read_csv('history.csv')

#plt.plot(h['loss'])
#plt.plot(h['val_loss'])