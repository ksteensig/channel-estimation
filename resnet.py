import tensorflow as tf
"""
class Residual(tf.keras.Model):  #@save
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv1D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None

        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv1D(
                num_channels, kernel_size=1, strides=strides)

        self.bn1 = tf.keras.layers.LayerNormalization()
        self.bn2 = tf.keras.layers.LayerNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y = tf.keras.layers.Dropout(0.1)(Y)
        Y += X
        return tf.keras.activations.relu(Y)

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
"""

class Residual(tf.keras.Model):  #@save
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = tf.keras.layers.Dense(num_channels)
        self.conv2 = tf.keras.layers.Dense(num_channels)
        self.conv3 = None
        
        self.bn1 = tf.keras.layers.LayerNormalization()
        self.bn2 = tf.keras.layers.LayerNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y = tf.keras.layers.Dropout(0.1)(Y)
        Y += X
        return tf.keras.activations.relu(Y)

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