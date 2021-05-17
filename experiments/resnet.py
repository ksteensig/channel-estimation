import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers

class Residual(tf.keras.Model):  #@save
    def __init__(self, num_channels):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units = num_channels,
                                        kernel_initializer=initializers.RandomNormal(stddev=1/num_channels**2),
                                        bias_initializer=initializers.Zeros())
        self.W2 = tf.keras.layers.Dense(units = num_channels,
                                        kernel_initializer=initializers.RandomNormal(stddev=1/num_channels**2),
                                        bias_initializer=initializers.Zeros())
        
        self.lnorm = tf.keras.layers.LayerNormalization()

    def call(self, X):
        print(X.shape)
        ResX = relu(self.W1(X))
        ResX = self.W2(ResX)
        ResX = Dropout(0.1)(ResX)
        
        return self.lnorm(X + ResX)

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X