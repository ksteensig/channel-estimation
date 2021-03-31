import tensorflow.keras as keras
import numpy as np

def compute_acc(y_true, y_pred, k):
    acc = keras.metrics.TopKCategoricalAccuracy(k=k)
    acc.update_state(y_true, y_pred)
    
    return acc.result().numpy()

def compute_pos_acc(y_true, y_pred):
    acc_ = 0
    
    for i in range(len(y_true)):
        idx_tp = np.where(y_true[i] == 1)[0]
        idx_tn = np.where(y_true[i] == 0)[0]
        r = np.sum(y_pred[i][idx_tp]) + np.sum(np.abs(y_pred[i][idx_tn] - 1))
        r = r/180
        
        acc_ = acc_ + r
        
    return acc_/len(y_true)

def compute_neg_acc(y_true, y_pred):
    acc_ = 0
    
    y_true = y_true.copy()
    y_true = np.abs(y_true - 1)
    
    y_pred = y_pred.copy()
    y_pred = np.abs(y_pred - 1)
    
    for i in range(len(y_true)):
        idx = np.where(y_true[i] == 1)[0]
        r = np.sum(y_pred[i][idx])/np.sum(y_true[i][idx])
        acc_ = acc_ + r
        
    return acc_/len(y_true)

def compute_fp(y_true, y_pred):
    fp = keras.metrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    
    return fp.result().numpy()/samples

def compute_fn(y_true, y_pred):
    fn = keras.metrics.FalseNegatives()
    fn.update_state(y_true, y_pred)
    
    return fn.result().numpy()/samples