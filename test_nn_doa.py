import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import sqrt,pi
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import data_generation as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]


testing_size = 20000

def data_initialization(N, K, L, freq):
    testing = f"data/testing_{N}_{K}_{L}"
    if not dg.check_data_exists(testing):
        labels, data = dg.generate_bulk_data(testing_size, N, K, L, freq)
        dg.save_generated_data(testing, labels, data)
        
    testing_labels, testing_data = dg.load_generated_data(testing)

    return testing_labels, testing_data
# antennas

N = 128

# users
K = 8

# frequency
freq = 1e9

# bits
L = 16

testing_labels, testing_data = data_initialization(N, K, L, freq)

model = load_model('models/users_8_snr_30_30_sgd_lr_0.1_momentum_0.5')

mse_snr = []

for i in [5,10,15,20,25,30]:
    snr = [i,i]
    print(snr)
    wgn_testing_data = dg.apply_wgn(testing_data, snr)

    norm_testing_labels, norm_testing_data = dg.normalize(testing_labels, wgn_testing_data)
    test_loss, mse = model.evaluate(norm_testing_data, norm_testing_labels, verbose=2)
    
    mse_snr.append((sqrt(mse)*pi)**2)

    print('\n MSE:', (sqrt(mse)*pi)**2)
    
    
plt.xlabel('SNR (dB)')
plt.ylabel('MSE')
plt.ylim(0.01, 0.5)
plt.semilogy([5,10,15,20,25,30], mse_snr)
plt.grid()
plt.show()