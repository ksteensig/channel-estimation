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
from sklearn.metrics import classification_report

from sklearn.metrics import mean_squared_error

from metrics import *

training_size = 200000
validation_size = 0.1 # 10% of training size

# receiver antennas
N = 16

# bits
L = 16

K = 8

# snr between 5 and 30 dB
snr = [5, 30]

learning_rate = 0.001

resolution = 180
samples = 5000

mse = {
            'cbn cov' : [],
            'cbn multi-vec' : [],
            'cbn resnet' : [],
            'cbn row cov' :  [],
            'rbn cov' : [],
            'rbn single-vec' : [],                    
            }

acc = {
        'cbn cov' : [],
        'cbn multi-vec' : [],
            'cbn resnet' : [],
            'cbn row cov' :  [],
        }

precision = {
        'cbn cov' : [],
        'cbn multi-vec' : [],
            'cbn resnet' : [],
            'cbn row cov' :  [],
        }

recall = {
        'cbn cov' : [],
        'cbn multi-vec' : [],
            'cbn resnet' : [],
            'cbn row cov' :  [],
        }



snrs = [5,10,15,20,25,30]

cutoff = 0.5
threshold_vec = cutoff

def cbn_cov(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = data / np.max(np.abs(data), axis=1).reshape((samples,1))
        
    pred = model.predict(data)
        
    pred_conv = np.zeros((len(labels), K))
    labels_conv = np.zeros((len(labels), K))
    
    
    for i in range(len(pred)):
        n = int(np.sum(labels[i]))
        pred_theta = (-pred[i]).argsort()[:n].copy()        
        pred_theta.sort()
        pred_conv[i][:n] = pred_theta / 180 #* np.pi - np.pi/2
        
        pred[i][pred[i] < cutoff] = 0
        pred[i][pred[i] >= cutoff] = 1
        
        temp = (-labels[i]).argsort()[:n].copy()
        temp.sort()
        labels_conv[i][:n] = temp / 180 #* np.pi - np.pi/2
    
    #compute_pos_acc(labels, pred, K)
    p = tf.keras.metrics.Precision(thresholds=threshold_vec)
    p.update_state(labels, pred)
    prec = p.result().numpy()
    
    r = tf.keras.metrics.Recall(thresholds=threshold_vec)
    r.update_state(labels, pred)
    rec = r.result().numpy()

    m = mean_squared_error(labels_conv, pred_conv) / mean_squared_error(labels_conv, np.zeros(labels_conv.shape))
        
    mse['cbn cov'].append(m)
    precision['cbn cov'].append(prec)
    recall['cbn cov'].append(rec)

def cbn_row_cov(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data[:, list(range(0,N))+list(range(-1,-(N+1), -1))]
    data = data / np.max(np.abs(data), axis=1).reshape((samples,1))
        
    pred = model.predict(data)
                
    pred_conv = np.zeros((len(labels), K))
    labels_conv = np.zeros((len(labels), K))
    
    
    for i in range(len(pred)):
        n = int(np.sum(labels[i]))
        pred_theta = (-pred[i]).argsort()[:n].copy()        
        pred_theta.sort()
        pred_conv[i][:n] = pred_theta / 180 #* np.pi - np.pi/2
        
        pred[i][pred[i] < cutoff] = 0
        pred[i][pred[i] >= cutoff] = 1
        
        temp = (-labels[i]).argsort()[:n].copy()
        temp.sort()
        labels_conv[i][:n] = temp / 180 #* np.pi - np.pi/2
        
    
    #acc_pos_ = compute_pos_acc(labels, pred, K)
    p = tf.keras.metrics.Precision(thresholds=threshold_vec)
    p.update_state(labels, pred)
    prec = p.result().numpy()
    
    r = tf.keras.metrics.Recall(thresholds=threshold_vec)
    r.update_state(labels, pred)
    rec = r.result().numpy()
    
    m = mean_squared_error(labels_conv, pred_conv) / mean_squared_error(labels_conv, np.zeros(labels_conv.shape))
        
    mse['cbn row cov'].append(m)
    precision['cbn row cov'].append(prec)
    recall['cbn row cov'].append(rec)


def cbn_resnet(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = data / np.max(np.abs(data), axis=1).reshape((samples,1))
        
    pred = model.predict(data)
            
    pred_conv = np.zeros((len(labels), K))
    labels_conv = np.zeros((len(labels), K))
    
        
    for i in range(len(pred)):
        n = int(np.sum(labels[i]))
        pred_theta = (-pred[i]).argsort()[:n].copy()        
        pred_theta.sort()
        pred_conv[i][:n] = pred_theta / 180 #* np.pi - np.pi/2
        
        pred[i][pred[i] < cutoff] = 0
        pred[i][pred[i] >= cutoff] = 1
        
        temp = (-labels[i]).argsort()[:n].copy()
        temp.sort()
        labels_conv[i][:n] = temp / 180 #* np.pi - np.pi/2
        
    
    #acc_pos_ = compute_pos_acc(labels, pred, K)
    p = tf.keras.metrics.Precision(thresholds=threshold_vec)
    p.update_state(labels, pred)
    prec = p.result().numpy()
    
    r = tf.keras.metrics.Recall(thresholds=threshold_vec)
    r.update_state(labels, pred)
    rec = r.result().numpy()
    
    m = mean_squared_error(labels_conv, pred_conv) / mean_squared_error(labels_conv, np.zeros(labels_conv.shape))
        
    mse['cbn resnet'].append(m)
    precision['cbn resnet'].append(prec)
    recall['cbn resnet'].append(rec)

def cbn_received(model, data_, labels_, snr):
    labels = np.copy(labels_)
    data = np.copy(data_)
    data = cbn_dg.apply_wgn(data, L, snr)
    
    data = np.concatenate((data.real,data.imag), axis=1)
    data = data.reshape((samples, 2*N*L))
    data = data / np.max(np.abs(data), axis=1).reshape((samples,1))
          
    pred = model.predict(data)
        
    pred_conv = np.zeros((len(labels), K))
    labels_conv = np.zeros((len(labels), K))
    
        
    for i in range(len(pred)):
        n = int(np.sum(labels[i]))
        pred_theta = (-pred[i]).argsort()[:n].copy()        
        pred_theta.sort()
        pred_conv[i][:n] = pred_theta / 180 #* np.pi - np.pi/2
        
        pred[i][pred[i] < cutoff] = 0
        pred[i][pred[i] >= cutoff] = 1
        
        temp = (-labels[i]).argsort()[:n].copy()
        temp.sort()
        labels_conv[i][:n] = temp / 180 #* np.pi - np.pi/2
    
    #acc_pos_ = compute_pos_acc(labels, pred, K)
    p = tf.keras.metrics.Precision(thresholds=threshold_vec)
    p.update_state(labels, pred)
    prec = p.result().numpy()
    
    r = tf.keras.metrics.Recall(thresholds=threshold_vec)
    r.update_state(labels, pred)
    rec = r.result().numpy()
    
    m = mean_squared_error(labels_conv, pred_conv) / mean_squared_error(labels_conv, np.zeros(labels_conv.shape))
        
    mse['cbn multi-vec'].append(m)
    precision['cbn multi-vec'].append(prec)
    recall['cbn multi-vec'].append(rec)


def rbn_received(model, data_, labels_, snr):
    labels = np.copy(labels_)
    labels = np.repeat(labels, L, axis=0).reshape(samples*L,K)
    data = np.copy(data_)
    
    data = cbn_dg.apply_wgn(data, L, snr)
    data = np.concatenate((data.real,data.imag), axis=1)
    data = data / np.max(np.abs(data), axis=1).reshape((samples*L,1))

    pred = model.predict(data)#*np.pi - np.pi/2
    
    #print(pred[0])
        
    labels = (labels + np.pi/2)/np.pi
    
    m = mean_squared_error(labels, pred) / mean_squared_error(labels, np.zeros(labels.shape))
    
    mse['rbn single-vec'].append(m)
    
def rbn_cov(model, data_, labels_, snr):
    labels = labels_.copy()
    data = data_.copy()
    
    data = cbn_dg.apply_wgn(data, L, snr).reshape((samples, L, N))
    data = cbn_dg.compute_cov(data)/L
    data = data / np.max(np.abs(data), axis=1).reshape((samples,1))
    
    pred = model.predict(data) #- np.pi/2
        
    labels = (labels + np.pi/2)/np.pi
    
    #print(labels[0])
    #print(pred[0])
    
    m = mean_squared_error(labels, pred) / mean_squared_error(labels, np.zeros(labels.shape))
   
    #m = np.linalg.norm(labels - pred)**2 / np.linalg.norm(labels)**2
    #print(m)
    #m = np.mean(m)

    mse['rbn cov'].append(m)
    