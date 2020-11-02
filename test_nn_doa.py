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

def normalize_add_wgn(labels, data, snr):
    data = dg.apply_wgn(data, snr)

    labels, data = dg.normalize(labels, data)

    return labels, data


denormalize = lambda x: (sqrt(x)*pi)**2

# antennas
N = 128

# users
K = [4,8,16,32]

# frequency
freq = 1e9

# bits
L = 16

snr = [5, 30]

learning_rate = 0.1
momentum = 0.5

plt.xlabel('SNR (dB)')
plt.ylabel('MSE')
plt.ylim(0.01, 0.5)

for k in K:
    model = load_model(f"models/users_{k}_sgd_lr_{learning_rate}_momentum_{momentum}")
    testing_labels, testing_data = data_initialization(N, k, L, freq)
    mse_snr = []
    for i in [5,10,15,20,25,30]:
        snr = [i,i]
        
        norm_testing_labels, norm_testing_data = normalize_add_wgn(testing_labels, testing_data, snr)
        test_loss, mse = model.evaluate(norm_testing_data, norm_testing_labels, verbose=2)
        
        mse_snr.append(denormalize(mse))

    plt.semilogy([5,10,15,20,25,30], mse_snr)
    
plt.legend(['K=4', 'K=8', 'K=16', 'K=32'])
plt.savefig(f"figures/performance_sgd_lr_{learning_rate}_momentum_{momentum}.svg", format='svg')