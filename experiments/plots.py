import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cbn_datagen as cbn_dg
import cbn_recv_datagen as cbn_recv_dg
import rbn_datagen as rbn_dg
import rbn_cov_datagen as rbn_cov_dg
import cbn_ae_datagen as cbn_ae_dg

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import tensorflow as tf
import pickle
import numpy as np
import scipy
from scipy.signal import find_peaks
from sklearn.metrics import classification_report

training_size = 200000
validation_size = 0.1 # 10% of training size

# receiver antennas
N = 16

# bits
L = 16

# snr between 5 and 30 dB
snr = [5, 30]

learning_rate = 0.001

resolution = 180


samples = 2000

def compute_acc(y_true, y_pred, k):
    acc = keras.metrics.TopKCategoricalAccuracy(k=k)
    acc.update_state(y_true, y_pred)
    
    return acc.result().numpy()

def compute_fp(y_true, y_pred):
    fp = keras.metrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    
    return fp.result().numpy()/samples

def compute_fn(y_true, y_pred):
    fn = keras.metrics.FalseNegatives()
    fn.update_state(y_true, y_pred)
    
    return fn.result().numpy()/samples

mse = {
            'cbn covariance' : [],
            'rbn covariance' : [],
            'cbn received' : [],
            'rbn received' : [],
            'cbn resnet' : [],
            'cbn row' :  [],
            'cbn autoencoder': []
            }
    
acc = {
        'cbn covariance' : [],
        'cbn received' : [],
            'cbn resnet' : [],
            'cbn row' :  [],
            'cbn autoencoder': []
        }
    
fp = {
            'cbn covariance' : [],
            'cbn received' : [],
            'cbn resnet' : [],
            'cbn row' :  [],
            'cbn autoencoder': []
            }
    
fn = {
            'cbn covariance' : [],
            'cbn received' : [],
            'cbn resnet' : [],
            'cbn row' :  [],
            'cbn autoencoder': []
            }

snrs = [5,10,15,20,25,30]

for k in [4]:
    cbn_model = load_model(f"models/CBN_N={N}_K={k}_L={L}")
    cbn_ae_model = load_model(f"models/CBN_ae_N={N}_K={k}_L={L}")
    cbn_row_model = load_model(f"models/CBN_row_N={N}_K={k}_L={L}")
    cbn_resnet_model = load_model(f"models/CBN_resnet_N={N}_K={k}_L={L}")
    cbn_recv_model = load_model(f"models/CBN_recv_N={N}_K={k}_L={L}")
    rbn_model = load_model(f"models/RBN_N={N}_K={k}_L={L}")
    rbn_cov_model = load_model(f"models/RBN_cov_N={N}_K={k}_L={L}")
    
    cbn_labels_, cbn_data_ = cbn_dg.generate_bulk_data(samples, N, k, L)
    cbn_ae_labels_, cbn_ae_data_ = cbn_ae_dg.generate_bulk_data(samples, N, k, L)
    cbn_row_labels_, cbn_row_data_ = cbn_dg.generate_bulk_data(samples, N, k, L)
    cbn_resnet_labels_, cbn_resnet_data_ = cbn_dg.generate_bulk_data(samples, N, k, L)
    cbn_recv_labels_, cbn_recv_data_ = cbn_recv_dg.generate_bulk_data(samples, N, k, L)
    rbn_labels_, rbn_data_ = rbn_dg.generate_bulk_data(samples, N, k, L)
    rbn_cov_labels_, rbn_cov_data_ = rbn_cov_dg.generate_bulk_data(samples, N, k, L)
    
    for s in snrs:
        snr = [s, s]
        
        cbn_labels = np.copy(cbn_labels_)
        cbn_data = np.copy(cbn_data_)
        cbn_data = cbn_dg.apply_wgn(cbn_data, L, snr).reshape((samples, L, N))
        cbn_data = cbn_dg.compute_cov(cbn_data)/L
        cbn_data = cbn_dg.normalize(cbn_data, snr)
        
        cbn_pred = cbn_model.predict(cbn_data)
        
        cbn_pred = cbn_pred / np.max(cbn_pred, axis=1).reshape(samples,1)
        
        for i in range(len(cbn_pred)):
            idx = find_peaks(cbn_pred[i], 0.05)[0]
            cbn_pred[i][:] = 0
            cbn_pred[i][idx] = 1
            
        cbn_row_labels = np.copy(cbn_row_labels_)
        cbn_data = np.copy(cbn_row_data_)
        cbn_data = cbn_dg.apply_wgn(cbn_data, L, snr).reshape((samples, L, N))
        cbn_data = cbn_dg.compute_cov(cbn_data)/L
        cbn_data = cbn_dg.normalize(cbn_data, snr)
        cbn_data = cbn_data[:, list(range(0,N))+list(range(-1,-(N+1), -1))]
        
        cbn_row_pred = cbn_row_model.predict(cbn_data)
        
        cbn_row_pred = cbn_row_pred / np.max(cbn_row_pred, axis=1).reshape(samples,1)
        
        for i in range(len(cbn_pred)):
            idx = find_peaks(cbn_row_pred[i], 0.05)[0]
            cbn_row_pred[i][:] = 0
            cbn_row_pred[i][idx] = 1
            
        cbn_resnet_labels = np.copy(cbn_resnet_labels_)
        cbn_data = np.copy(cbn_resnet_data_)
        cbn_data = cbn_dg.apply_wgn(cbn_data, L, snr).reshape((samples, L, N))
        cbn_data = cbn_dg.compute_cov(cbn_data)/L
        cbn_data = cbn_dg.normalize(cbn_data, snr)
        
        cbn_resnet_pred = cbn_model.predict(cbn_data)
        
        cbn_resnet_pred = cbn_resnet_pred / np.max(cbn_resnet_pred, axis=1).reshape(samples,1)
        
        for i in range(len(cbn_pred)):
            idx = find_peaks(cbn_resnet_pred[i], 0.05)[0]
            cbn_resnet_pred[i][:] = 0
            cbn_resnet_pred[i][idx] = 1
            
        
        cbn_recv_labels = cbn_recv_labels_.copy()
        cbn_recv_data = cbn_recv_data_.copy()
        cbn_recv_data = cbn_recv_dg.normalize_add_wgn(cbn_recv_data, L, snr)
        cbn_recv_pred = cbn_recv_model.predict(cbn_recv_data)
        
        
        cbn_recv_pred = cbn_recv_pred / np.max(cbn_recv_pred, axis=1).reshape(samples*L,1)
        
        for i in range(len(cbn_recv_pred)):
            idx = find_peaks(cbn_recv_pred[i], 0.05)[0]
            cbn_recv_pred[i][:] = 0
            cbn_recv_pred[i][idx] = 1
        
        
        cbn_ae_labels = cbn_ae_labels_.copy()
        cbn_ae_data = cbn_ae_data_.copy()
                
        cbn_ae_data = cbn_ae_dg.apply_wgn(cbn_ae_data, L, snr)
        
        cbn_ae_data = cbn_ae_data.reshape(samples, N*L)
        cbn_ae_data = np.concatenate((cbn_ae_data.real,cbn_ae_data.imag), axis=1)
        cbn_ae_data = cbn_ae_data / np.max(np.abs(cbn_ae_data), axis=1).reshape((samples,1))
        cbn_ae_pred = cbn_ae_model.predict(cbn_ae_data)
        
        cbn_ae_pred = cbn_ae_pred / np.max(cbn_ae_pred, axis=1).reshape(samples,1)
        
        for i in range(len(cbn_ae_pred)):
            idx = find_peaks(cbn_ae_pred[i], 0.05)[0]
            cbn_ae_pred[i][:] = 0
            cbn_ae_pred[i][idx] = 1
        
        rbn_labels = rbn_labels_.copy()
        rbn_data = rbn_data_.copy()
        rbn_dg.normalize_add_wgn(rbn_labels, rbn_data, L, snr)
        rbn_pred = rbn_model.predict(rbn_data)*np.pi - np.pi/2
        
        rbn_labels = rbn_labels * np.pi - np.pi/2
        
        rbn_cov_labels = rbn_cov_labels_.copy()
        rbn_cov_data = rbn_cov_data_.copy()
        rbn_cov_dg.normalize_add_wgn(rbn_cov_labels, rbn_cov_data, snr)
        rbn_cov_pred = rbn_cov_model.predict(rbn_cov_data)*np.pi - np.pi/2
        
        rbn_cov_labels = rbn_cov_labels * np.pi - np.pi/2
        
        mse['cbn covariance'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(cbn_labels.T, cbn_pred.T)).numpy())
        mse['rbn covariance'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(rbn_cov_labels.T, rbn_cov_pred.T)).numpy())
        mse['cbn received'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(cbn_recv_labels.T, cbn_recv_pred.T)).numpy())
        mse['rbn received'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(rbn_labels.T, rbn_pred.T)).numpy())
        mse['cbn row'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(cbn_row_labels.T, cbn_row_pred.T)).numpy())
        mse['cbn resnet'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(cbn_resnet_labels.T, cbn_resnet_pred.T)).numpy())
        mse['cbn autoencoder'].append(tf.reduce_mean(tf.keras.losses.mean_squared_error(cbn_ae_labels.T, cbn_ae_pred.T)).numpy())
        
        acc['cbn covariance'].append(compute_acc(cbn_labels, cbn_pred, k))
        acc['cbn received'].append(compute_acc(cbn_recv_labels, cbn_recv_pred, k))
        acc['cbn row'].append(compute_acc(cbn_row_labels, cbn_row_pred, k))
        acc['cbn resnet'].append(compute_acc(cbn_resnet_labels, cbn_resnet_pred, k))
        acc['cbn autoencoder'].append(compute_acc(cbn_ae_labels, cbn_ae_pred, k))
        
        fp['cbn covariance'].append(compute_fp(cbn_labels, cbn_pred))
        fp['cbn received'].append(compute_fp(cbn_recv_labels, cbn_recv_pred)/L)
        fp['cbn row'].append(compute_fp(cbn_row_labels, cbn_row_pred))
        fp['cbn resnet'].append(compute_fp(cbn_resnet_labels, cbn_resnet_pred))
        fp['cbn autoencoder'].append(compute_fp(cbn_ae_labels, cbn_ae_pred))
        
        fn['cbn covariance'].append(compute_fn(cbn_labels, cbn_pred))
        fn['cbn received'].append(compute_fn(cbn_recv_labels, cbn_recv_pred)/L)
        fn['cbn row'].append(compute_fn(cbn_row_labels, cbn_row_pred))
        fn['cbn resnet'].append(compute_fn(cbn_resnet_labels, cbn_resnet_pred))
        fn['cbn autoencoder'].append(compute_fn(cbn_ae_labels, cbn_ae_pred))

    for i in list(mse.keys()):
        plt.semilogy(snrs, mse[i])
        mse[i].clear()

    plt.ylabel('MSE')
    plt.xlabel('SNR (dB)')
    plt.ylim([None, 1])
    plt.legend(list(mse.keys()))
    plt.savefig(f'figures/mse_N={N}_K={k}_L={L}.png')
    plt.clf()
    
    for i in list(acc.keys()):
        print(snrs, acc[i])
        plt.plot(snrs, acc[i])
        acc[i].clear()
        
    plt.ylabel('Accuracy')
    plt.xlabel('SNR (dB)')
    plt.legend(list(acc.keys()))
    plt.savefig(f'figures/acc_N={N}_K={k}_L={L}.png')
    plt.clf()
    
    for i in list(fp.keys()):
        plt.plot(snrs, fp[i])
        fp[i].clear()
        
    plt.ylabel('False positives')
    plt.xlabel('SNR (dB)')
    plt.legend(list(fp.keys()))
    plt.savefig(f'figures/fp_N={N}_K={k}_L={L}.png')
    plt.clf()
    
    for i in list(fn.keys()):
        plt.plot(snrs, fn[i])
        fn[i].clear()
        
    plt.ylabel('False negatives')
    plt.xlabel('SNR (dB)')
    plt.legend(list(fn.keys()))
    plt.savefig(f'figures/fn_N={N}_K={k}_L={L}.png')
    plt.clf()

    cbn_history = pickle.load(open(f"history/CBN_N={N}_K={k}_L={L}", 'rb'))
    cbn_recv_history = pickle.load(open(f"history/CBN_recv_N={N}_K={k}_L={L}", 'rb'))
    rbn_history = pickle.load(open(f"history/RBN_N={N}_K={k}_L={L}", 'rb'))
    rbn_cov_history = pickle.load(open(f"history/RBN_cov_N={N}_K={k}_L={L}", 'rb'))
    
    plt.plot(cbn_history['loss'])
    plt.plot(cbn_history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.xlabel('epoch')
    plt.ylabel('binary cross entropy')
    plt.savefig(f"figures/CBN_loss_N={N}_K={k}_L={L}.png", format='png')
    plt.clf()
    
    plt.plot(cbn_recv_history['loss'])
    plt.plot(cbn_recv_history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.xlabel('epoch')
    plt.ylabel('binary cross entropy')
    plt.savefig(f"figures/CBN_recv_loss_N={N}_K={k}_L={L}.png", format='png')
    plt.clf()
    
    plt.plot(rbn_history['loss'])
    plt.plot(rbn_history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.savefig(f"figures/RBN_loss_N={N}_K={k}_L={L}.png", format='png')
    plt.clf()
    
    plt.plot(rbn_cov_history['loss'])
    plt.plot(rbn_cov_history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.savefig(f"figures/RBN_cov_loss_N={N}_K={k}_L={L}.png", format='png')
    plt.clf()

"""
for k in [4, 8]:
    model = load_model(f"models/CBN_N={N}_K={k}_L={L}")
    
    
    history = pickle.load(open(f"history/CBN_N={N}_K={k}_L={L}", 'rb'))
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss', 'val_los'])
    plt.xlabel('epoch')
    plt.ylabel('binary cross entropy')
    plt.savefig(f"figures/CBN_loss_N={N}_K={k}_L={L}.png", format='png')
    
    plt.clf()
    
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig(f"figures/CBN_accuracy_N={N}_K={k}_L={L}.png", format='png')
    
    plt.clf()
    
    mse = []
    
    snrs = [5,10,15,20,25,30]
    
    for i in snrs:
        labels, data = cbn_dg.generate_bulk_data(2000, N, k, L, 2.4e9)
        cbn_dg.normalize_add_wgn(labels, data, [i,i])
        y_pred = model.predict(data)
        
        #y_pred = y_pred / np.max(y_pred)
                
        s = keras.losses.mean_squared_error(labels.T, y_pred.T)
        
        m = tf.reduce_mean(s)
        
        mse.append(m)
        
    plt.semilogy(snrs, mse)
    plt.xlabel('snr')
    plt.ylabel('mse')
    plt.savefig(f"figures/CBN_pred_N={N}_K={k}_L={L}.png", format='png')
    
    plt.clf()
"""