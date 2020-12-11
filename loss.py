import tensorflow as tf
import numpy as np

output_size = 180

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
    
    return tf.reduce_mean(tf.norm(ypred - f, axis=1)**2) #* 1/tf.size(f)