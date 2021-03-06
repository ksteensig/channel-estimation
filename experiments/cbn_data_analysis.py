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

import cbn_datagen as dg
import cbn_recv_datagen as dg_

print('TensorFlow Version:', tf.__version__)

# receiver antennas
N = 16

# users
K = 4

# frequency
freq = 2.4e9

# bits
L = 16

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]

learning_rate = 0.001

resolution = 180

labels, data = dg.data_initialization(1, N, K, L, freq, resolution, snr, cache=False)

data = dg.apply_wgn(data, L, snr).reshape((1, L, N))
    
data = dg.compute_cov(data)/L
    
data = dg.normalize(data, snr)

labels_, data_ = dg_.generate_bulk_data(1, N, K, L)
data_ = dg_.normalize_add_wgn(data_, L, [1000,1000])

data_ = data_[:, list(range(0,N))+list(range(-1,-(N+1), -1))]

model = load_model(f"models/CBN_N={N}_K={K}_L={L}")
model_ = load_model(f"models/CBN_row_N={N}_K={K}_L={L}")

prediction = model_.predict(data_)

res = prediction[0] / np.max(prediction[0])

plt.plot(labels_[0])
plt.plot(res)

peaks = find_peaks(res, 0.05)[0]

plt.plot(peaks, res[peaks], 'x')

#plt.show()

#h = pd.read_csv('history.csv')

#plt.plot(h['loss'])
#plt.plot(h['val_loss'])