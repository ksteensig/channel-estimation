import numpy as np
#from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle


# SNR is a range between min and max SNR in dB
def generate_signal(N = 16, K = 4, L = 16, f = 2.4e9, theta_bound = np.pi/2):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
        
    # antenna array
    array = np.linspace(0,N-1,N)*d/wl

    theta = np.random.rand(K,1) * np.pi - np.pi/2

    
    alpha = (np.random.randn(K,1) + 1j*np.random.randn(K,1))*np.sqrt(1/2)
        
    response = np.exp(-1j*2*np.pi*array*np.sin(theta))*np.sqrt(1/N)    

    Y = np.dot(response.T, alpha).repeat(L, axis=1)

                
    return theta, Y, alpha

rads_to_vector = lambda x, D: np.floor((x + np.pi/2)/np.pi * D).astype(int)

def compute_H(theta, N, f = 2.4e9):
    c = 3e8 # speed of light
    wl = c/f # wavelength (lambda)
    d = wl/2 # uniform distance between antennas
    array = np.linspace(0,N-1,N,dtype=np.float32)*d/wl
    
    array_response = np.exp(-1j*2*np.pi*array*np.sin(theta))*np.sqrt(1/N)  
    return array_response.T


def soft_block_thresh(Z, l, alpha, T):
    rmse = tf.norm(Z, axis=2)
    #print(rmse.shape)
    rmse = tf.tile(tf.expand_dims(rmse, axis=2), multiples=[1,1,T])
    #print(rmse.shape)
    return (1 - l/tf.maximum(rmse, l))*Z # / (1+alpha)


mul_X = lambda w, x: tf.transpose(tf.tensordot(w, x, axes=[[1],[1]]), [1,0,2])
mul_Y = lambda w, y: tf.transpose(tf.tensordot(w, y, axes=[[1],[1]]), [1,0,2])

#transform_X = lambda x: tf.transpose(tf.expand_dims(x, 2), [0,3,1,2])
#transform_hgX = lambda x: tf.squeeze(tf.transpose(x, [0,2,1,3]), axis=3)
#conv_X = lambda hg, x: transform_hgX(hg(transform_X(x)))
conv_X = lambda hg, x: hg(x)

class LISTA(tf.keras.Model):  #@save
    def __init__(self, num_iter, D, N, T):
        super(LISTA, self).__init__()
        self.num_iter = num_iter
        self.T = T
        self.N = N
        self.D = D
                
        search_space = np.linspace(0,D-1, D,dtype=np.float32) / D * np.pi - np.pi/2
        search_space = search_space.reshape((D, 1))
        
        H = compute_H(search_space, N)
        
        L = np.linalg.norm(H)**2
        lam_initial = 4e-3
        
        step_size = 0.5/L
        
        We_initial = 2 * step_size * H.T.conj()
        Wt_initial = np.eye(D) - H.T.conj().dot(H) * step_size
        
        self.Wt_r = tf.Variable(Wt_initial.real.copy(), dtype=tf.float32)
        self.Wt_i = tf.Variable(Wt_initial.imag.copy(), dtype=tf.float32)
        
        self.We_r = tf.Variable(We_initial.real.copy(), dtype=tf.float32)
        self.We_i = tf.Variable(We_initial.imag.copy(), dtype=tf.float32)
        
        self.lam_list = []
        
        for i in range(num_iter):
            self.lam_list.append(tf.Variable(lam_initial, dtype=tf.float32))
    
        self.alpha = tf.Variable(0.1, dtype=tf.float32)

    def call(self, Y):
        Y_r = tf.math.real(Y)
        Y_i = tf.math.imag(Y)
        
                
        X_r = mul_Y(self.We_r, Y_r)
        X_r = soft_block_thresh(X_r, self.lam_list[0], self.alpha, self.T)
        X_i = mul_Y(self.We_i, Y_i)
        X_i = soft_block_thresh(X_i, self.lam_list[0], self.alpha, self.T)

        for i in range(1, self.num_iter):
            X_r = mul_X(self.Wt_r, X_r) - mul_X(self.Wt_i, X_i) + mul_Y(self.We_r, Y_r) - mul_Y(self.We_i, Y_i)
            X_r = soft_block_thresh(X_r, self.lam_list[i], self.alpha, self.T)
            X_i = mul_X(self.Wt_r, X_i) + mul_X(self.Wt_i, X_r) + mul_Y(self.We_r, Y_i) + mul_Y(self.We_i, Y_r)
            X_i = soft_block_thresh(X_i, self.lam_list[i], self.alpha, self.T)
            
        X = tf.concat([X_r, X_i], axis=1)
        
        return X #self.bn(X)

class LISTA_Toeplitz(tf.keras.Model):  #@save
    def __init__(self, num_iter, D, N, T):
        super(LISTA_Toeplitz, self).__init__()
        self.num_iter = num_iter
        self.T = T
        self.N = N
        self.D = D
        
        search_space = np.linspace(0,D-1, D) / D * np.pi - np.pi/2
        search_space = search_space.reshape((D, 1))
        
        H = compute_H(search_space, N)
        
        L = np.linalg.norm(H)**2
        lam_initial = 4e-3
        
        step_size = 0.5/L
        
        We_initial = 2 * step_size * H.T.conj()
        
        self.hg_r = tf.keras.layers.Conv1D(T, 2*D-1, activation='linear', padding='same', use_bias=False)
        self.hg_i = tf.keras.layers.Conv1D(T, 2*D-1, activation='linear', padding='same', use_bias=False)
        
        #self.hg_r = tf.keras.layers.TimeDistributed(self.hg_r_)
        #self.hg_i = tf.keras.layers.TimeDistributed(self.hg_i_)
        
        self.We_r = tf.Variable(We_initial.real.copy(), dtype=tf.float32)
        self.We_i = tf.Variable(We_initial.imag.copy(), dtype=tf.float32)
        
        self.lam_list = []
        
        for i in range(num_iter):
            self.lam_list.append(tf.Variable(lam_initial, dtype=tf.float32))
    
        self.alpha = tf.Variable(0.1, dtype=tf.float32)

    def call(self, Y):
        
        Y_r = tf.math.real(Y)
        Y_i = tf.math.imag(Y)

        X_r = mul_Y(self.We_r, Y_r)
        X_r = soft_block_thresh(X_r, self.lam_list[0], self.alpha, self.T)
        X_i = mul_Y(self.We_i, Y_i)
        X_i = soft_block_thresh(X_i, self.lam_list[0], self.alpha, self.T)
        

        for i in range(1, self.num_iter):
            X_r = conv_X(self.hg_r, X_r) - conv_X(self.hg_i, X_i) + mul_Y(self.We_r, Y_r) - mul_Y(self.We_i, Y_i)
            X_r = soft_block_thresh(X_r, self.lam_list[i], self.alpha, self.T)
            X_i = conv_X(self.hg_r, X_i) + conv_X(self.hg_i, X_r) + mul_Y(self.We_r, Y_i) + mul_Y(self.We_i, Y_r)
            X_i = soft_block_thresh(X_i, self.lam_list[i], self.alpha, self.T)
        
        X = tf.concat([X_r, X_i], axis=1)
        
        return X

def data_generation(N, K, T, D, samples):
    Y = np.zeros((samples, N, T), dtype=np.complex64)
    Theta = np.zeros((samples, K))
    Alpha = np.zeros((samples, K), dtype=np.complex64)
    
    labels = np.zeros((samples, D, T), dtype=np.complex64)
    
    for i in range(samples):
        theta, Yi, alpha = generate_signal(N, K, T)
        Y[i] = Yi
        Theta[i] = theta.flatten()
        
        idx = rads_to_vector(theta, D)
        labels[i, idx, :] = np.repeat(alpha, repeats=T, axis=0).reshape((K,1,T))
    
    #Y = Y.reshape((samplesT, N_max))
    #labels = labels.reshape((samples, D, T))
        
    labels = np.concatenate([labels.real, labels.imag], axis=1)
    #labels = labels.transpose((0,2,1)) #/ np.max(np.abs(labels))
    data = Y
    
    return data, labels

def add_noise(data, T, N, SNR, samples):    
    db2pow = 10**(np.random.uniform(SNR[0], SNR[1], size=(samples,1))/10)
    
    db2pow = tf.expand_dims(db2pow, axis=-1)
    db2pow = tf.tile(db2pow, multiples=[1,N,T])
    
    data = data + np.random.randn(*data.shape)*np.sqrt(0.5/db2pow) + 1j*np.random.randn(*data.shape)*np.sqrt(0.5/db2pow)
    
    data = data.astype(dtype=np.complex64)
    
    return data

def train_lista(num_iter, data, labels, N, K, T, D):
    lista_model = LISTA(num_iter, D, N, T)

    learning_rate = 0.001
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))
    
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-5)
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    
    lista_model.compile(optimizer=adam, loss='mse')
    
    m = lista_model.fit(data, labels, batch_size=32, epochs=200, validation_split=0.1, callbacks=[stopping, lrate])
    
    with open(f"history/LISTA_N={N}_K={K}_T={T}", 'wb') as f:
        pickle.dump(m.history, f)

    lista_model.save(f"models/LISTA_N={N}_K={K}_T={T}")

def train_lista_toeplitz(num_iter, data, labels, N, K, T, D):
    lista_model = LISTA_Toeplitz(num_iter, D, N, T)

    learning_rate = 0.001
    
    adaptive_learning_rate = lambda epoch: learning_rate/(2**np.floor(epoch/10))
    
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-5)
    lrate = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)
    
    lista_model.compile(optimizer=adam, loss='mse')
    
    m = lista_model.fit(data, labels, batch_size=32, epochs=200, validation_split=0.1, callbacks=[stopping, lrate])
    
    with open(f"history/LISTA_Toeplitz_N={N}_K={K}_T={T}", 'wb') as f:
        pickle.dump(m.history, f)

    lista_model.save(f"models/LISTA_Toeplitz_N={N}_K={K}_T={T}")
    
    