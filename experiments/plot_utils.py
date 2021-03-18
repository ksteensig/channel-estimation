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

from metrics import *

training_size = 200000
validation_size = 0.1 # 10% of training size

# receiver antennas
N = 16

# bits
L = 16

K = 4 # 8

# snr between 5 and 30 dB
snr = [5, 30]

learning_rate = 0.001

resolution = 180
samples = 5000

mse = {
            'cbn cov' : [],
            'cbn autoenc': []
            
            #'rbn cov' : [],
            #'cbn received' : [],
            #'rbn received' : [],
            #'cbn resnet' : [],
            #'cbn row cov' :  [],
                    
            }
    
acc = {
        'cbn cov' : [],
        #'cbn received' : [],
            #'cbn resnet' : [],
            #'cbn row cov' :  [],
            'cbn autoenc': []
        }
    
fp = {
            'cbn cov' : [],
            #'cbn received' : [],
            #'cbn resnet' : [],
            #'cbn row cov' :  [],
            'cbn autoenc': []
            }
    
fn = {
            'cbn cov' : [],
            #'cbn received' : [],
            #'cbn resnet' : [],
            #'cbn row cov' :  [],
            'cbn autoenc': []
            }

snrs = [5,10,15,20,25,30]

peak_cut = 0.05

def cbn_cov(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = cbn_dg.normalize(data, snr)
        
    pred = model.predict(data)
        
    pred = pred / np.max(pred, axis=1).reshape(samples,1)
        
    for i in range(len(pred)):
        idx = find_peaks(pred[i], peak_cut)[0]
        pred[i][:] = 0
        pred[i][idx] = 1
        
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    acc_ = compute_acc(labels, pred, K)
    fp_ = compute_fp(labels, pred)
    fn_ = compute_fn(labels, pred)
        
    mse['cbn cov'].append(mse_)
    acc['cbn cov'].append(acc_)
    fp['cbn cov'].append(fp_)
    fn['cbn cov'].append(fn_)

def cbn_row_cov(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = cbn_dg.normalize(data, snr)
    data[:, list(range(0,N))+list(range(-1,-(N+1), -1))]
        
    pred = model.predict(data)
        
    pred = pred / np.max(pred, axis=1).reshape(samples,1)
        
    for i in range(len(pred)):
        idx = find_peaks(pred[i], peak_cut)[0]
        pred[i][:] = 0
        pred[i][idx] = 1
        
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    acc_ = compute_acc(labels, pred, K)
    fp_ = compute_fp(labels, pred)
    fn_ = compute_fn(labels, pred)
        
    mse['cbn row cov'].append(mse_)
    acc['cbn row cov'].append(acc_)
    fp['cbn row cov'].append(fp_)
    fn['cbn row cov'].append(fn_)


def cbn_resnet(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = cbn_dg.normalize(data, snr)
        
    pred = model.predict(data)
        
    pred = pred / np.max(pred, axis=1).reshape(samples,1)
        
    for i in range(len(pred)):
        idx = find_peaks(pred[i], peak_cut)[0]
        pred[i][:] = 0
        pred[i][idx] = 1
        
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    acc_ = compute_acc(labels, pred, K)
    fp_ = compute_fp(labels, pred)
    fn_ = compute_fn(labels, pred)
    
    mse['cbn resnet'].append(mse_)
    acc['cbn resnet'].append(acc_)
    fp['cbn resnet'].append(fp_)
    fn['cbn resnet'].append(fn_)

def cbn_received(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_recv_dg.normalize_add_wgn(data, L, snr)
    
    data[:, list(range(0,N))+list(range(-1,-(N+1), -1))]
        
    pred = model.predict(data)
        
    pred = pred / np.max(pred, axis=1).reshape(samples*L,1)
        
    for i in range(len(pred)):
        idx = find_peaks(pred[i], peak_cut)[0]
        pred[i][:] = 0
        pred[i][idx] = 1
        
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    acc_ = compute_acc(labels, pred, K)
    fp_ = compute_fp(labels, pred) / L
    fn_ = compute_fn(labels, pred) / L
        
    mse['cbn received'].append(mse_)
    acc['cbn received'].append(acc_)
    fp['cbn received'].append(fp_)
    fn['cbn received'].append(fn_)

def cbn_autoenc(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    
    data = cbn_ae_dg.apply_wgn(data, L, snr).reshape((samples, L*N))
    data = np.concatenate((data.real,data.imag), axis=1)    
    data = data / np.max(np.abs(data), axis=1).reshape((samples, 1))

    pred = model.predict(data)
        
    pred = pred / np.max(pred, axis=1).reshape(samples,1)
        
    for i in range(len(pred)):
        idx = find_peaks(pred[i], peak_cut)[0]
        pred[i][:] = 0
        pred[i][idx] = 1
        
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    acc_ = compute_acc(labels, pred, K)
    fp_ = compute_fp(labels, pred)
    fn_ = compute_fn(labels, pred)
        
    mse['cbn autoenc'].append(mse_)
    acc['cbn autoenc'].append(acc_)
    fp['cbn autoenc'].append(fp_)
    fn['cbn autoenc'].append(fn_)

def rbn_received(model, data_, labels_, snr):
    labels = labels_.copy()
    data = data_.copy()
    
    rbn_dg.normalize_add_wgn(labels, data, L, snr)
    pred = model.predict(data)*np.pi - np.pi/2
        
    labels = labels * np.pi - np.pi/2
    
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    
    mse['rbn received'].append(mse_)
    
def rbn_cov(model, data_, labels_, snr):
    labels = labels_.copy()
    data = data_.copy()
    
    rbn_cov_dg.normalize_add_wgn(labels, data, snr)
    pred = model.predict(data)*np.pi - np.pi/2
        
    labels = labels * np.pi - np.pi/2
    
    mse_ = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels.T, pred.T)).numpy()
    
    mse['rbn cov'].append(mse_)
    