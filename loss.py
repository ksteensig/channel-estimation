import tensorflow as tf
import numpy as np

from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

output_size = 180

loss_lookup = np.zeros((output_size, output_size), dtype=np.float16)

for i in range(output_size):
    for j in range(output_size):
        if i==j:
            loss_lookup[i,j] = 0.8
        else:
            loss_lookup[i,j] =0.1*2**(-abs(i-j))

comparator = tf.constant(tf.ones([output_size], dtype=tf.float16), dtype=tf.float16)

@tf.function
def loss_fun_body(ytrue):
    condition = tf.equal(ytrue, comparator)
    indices = tf.where(condition)
    
    return tf.reduce_sum(tf.gather_nd(loss_lookup, indices), axis=0)

@tf.function
def loss_fun(ytrue, ypred):
    f = tf.map_fn(loss_fun_body, ytrue)
    
    #return tf.keras.losses.MSE(tf.transpose(f), tf.transpose(ypred))
    return tf.reduce_mean(tf.norm(ypred - f, axis=1)**2)