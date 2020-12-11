import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from resnet import Residual

import data_generation_v2 as dg

print('TensorFlow Version:', tf.__version__)

#settings = [()]

training_size = 50000
validation_size = 0.1 # 10% of training size

def data_initialization(training_size, N, K, L, freq, theta_dist = 'uniform'):
    training = f"data/training_v2_{N}_{K}_{L}"
    if not dg.check_data_exists(training):
    
        labels, data = dg.generate_bulk_data(training_size, N, K, L, freq, theta_dist)
        
        training_data = tf.convert_to_tensor(data)
        training_labels = tf.convert_to_tensor(labels)

        training_data = normalize_add_wgn(training_data, snr)
        
        C = tf.matmul(training_data, tf.transpose(training_data, perm=[0,2,1]))
        
        C = tf.reshape(C, [len(C),N*N])
        
        C = C/tf.reshape(tf.norm(C,axis=1), (-1, 1))
    
        r,i = tf.math.real(C), tf.math.imag(C)
    
        training_data = tf.cast(tf.concat([r, i], axis=1), dtype=tf.float32)
        
        dg.save_generated_data(training, training_labels, training_data)
        return training_labels, training_data
        
    labels, data = dg.load_generated_data(training)
    
    return labels, tf.cast(data, dtype=tf.float32)

# antennas
N = 8

# users
K = [6]

# frequency
freq = 1e9

# bits
L = 500

min_snr = 5
max_snr = 30

# snr between 5 and 30
snr = [min_snr, max_snr]


learning_rate = 0.1
epoch_warmup = 10
epoch_decay = 20
adaptive_learning_rate = lambda epoch: learning_rate #* min(min((epoch+1)/epoch_warmup, (epoch_decay/(epoch+1))**2), 1)
#momentum = 0.9

epochs = 100
batch_size = 800    
        
output_size = 180
block_depth = 15

loss_lookup = np.zeros((output_size, output_size), dtype=np.float32)

for i in range(output_size):
    for j in range(output_size):
        if i==j:
            loss_lookup[i,j] = 0.8
        else:
            loss_lookup[i,j] =0.1*2**(-abs(i-j))

comparator = tf.constant(tf.ones([output_size]))

def loss_fun_body(ytrue):
    condition = tf.equal(ytrue, comparator)
    indices = tf.where(condition)
    
    return tf.reduce_sum(tf.gather_nd(loss_lookup, indices), axis=0)


def loss_fun(ytrue, ypred):
    f = tf.map_fn(loss_fun_body, ytrue)
    
    print(f)
    
    return ypred - f #* 1/tf.size(f)

def apply_wgn(Y, SNR):
    shape = Y.get_shape()
    db2pow = 10**(np.random.uniform(SNR[0], SNR[1])/10)
    
    # N = [n1 n2 .. nL]
    N1 = tf.random.normal(shape)*np.sqrt(0.5/db2pow)
    N2 = tf.random.normal(shape)*np.sqrt(0.5/db2pow)
    
    return Y + tf.complex(N1, N2)

def normalize_add_wgn(data, snr):
    data = apply_wgn(data, snr)
    
    return data

# train neural network based on the covariance of the Y data
def train_model_v2(N, K, L, freq, snr):
    training_labels, training_data = data_initialization(training_size, N, K, L, freq)

    # define model
    input_ = tf.keras.layers.Input(shape=[2*N*N])
    
    h1 = tf.keras.layers.Dense(output_size)(input_)
    h1 = tf.keras.layers.Reshape((output_size,1))(h1)
    h1 = Residual(block_depth, output_size)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)
    """
    h1 = tf.keras.layers.Dense(2*output_size)(h1)
    h1 = tf.keras.layers.Reshape((2*output_size,1))(h1)
    h1 = Residual(block_depth, 2*output_size)(h1)
    h1 = tf.keras.layers.Dropout(0.5)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)
    
    h1 = tf.keras.layers.Dense(4*output_size)(h1)
    h1 = tf.keras.layers.Reshape((4*output_size,1))(h1)
    h1 = Residual(block_depth, 4*output_size)(h1)
    h1 = tf.keras.layers.Dropout(0.5)(h1)
    h1 = tf.keras.layers.MaxPooling1D()(h1)
    h1 = tf.keras.layers.Flatten()(h1)
    """
    output = tf.keras.layers.Dense(output_size, activation='sigmoid')(h1)
    
    model = keras.Model(inputs=[input_], outputs=[output] )

    sgd = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=sgd, loss=loss_fun)
    
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)
    m = model.fit(training_data, training_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_size, callbacks=[lrate, stopping])
    
    #tf.keras.utils.plot_model(
    #    model,
    #    to_file="model.png",
    #)

    model.save(f"models/dnn_v2_users_{K}_bits_{L}_sgd_lr_{learning_rate}")

    plt.plot(m.history['loss'], 'r')
    plt.plot(m.history['val_loss'], 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig(f"figures/learning_curve_users_{K}_bits_{L}_sgd_lr_{learning_rate}.svg", format='svg')
    plt.clf()
    
for k in K:
    train_model_v2(N, k, L, freq, snr)