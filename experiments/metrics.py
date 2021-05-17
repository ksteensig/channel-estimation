import tensorflow.keras as keras
import numpy as np

def compute_pos_acc(y_true, y_pred, K):
    acc_ = 0
    
    for i in range(len(y_true)):
        idx_tp = np.where(y_true[i] == 1)[0]
        idx_tn = np.where(y_true[i] == 0)[0]
        r = np.sum(y_pred[i][idx_tp]) + np.sum(np.abs(y_pred[i][idx_tn] - 1))
        r = r/180
        
        acc_ = acc_ + r
        
    return acc_/len(y_true)