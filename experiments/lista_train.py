from lista import data_generation, add_noise, train_lista, train_lista_toeplitz, LISTA
import tensorflow as tf
import numpy as np

#tf.config.experimental.enable_mlir_graph_optimization()

K_list = [4, 8]

T_max = 4

T_list = [1, T_max]

N = 64

D = 180

SNR = [5,30]

samples = 200000

num_iter = 3

for K in K_list:
    data,labels = data_generation(N, K, T_max, D, samples)
    data = add_noise(data, T_max, N, SNR, samples)

    
    for T in T_list:
        train_lista_toeplitz(num_iter, data[:,:,:T], labels[:,:,:T], N, K, T, D)
        train_lista(num_iter, data[:,:,:T], labels[:,:,:T], N, K, T, D)
        
        