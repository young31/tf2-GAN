import numpy as np
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input, Reshape, Concatenat
from tensorflow.keras.models import Model, Sequential, load_model

warnings.filterwarnings('ignore')
tf.config.experimental.set_visible_devices([], 'GPU') # if cannot find gpu device

class CGAN(keras.Model):
    def __init__(self, x_dim, y_dim, z_dim): # dim: tuple
        super(CGAN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def compile(self, g_optim, d_optim, loss_fn):
        super(CGAN, self).compile()
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.loss_fn = loss_fn
        
    def build_generator(self): 
        activation = 'relu'
        z = Input(shape = self.z_dim)
        y = Input(shape = self.y_dim)
        inputs = Concatenate()([z, y])
        
        h = Dense(256)(inputs)
        h = Activation(activation)(h)
        h = BatchNormalization(momentum=0.8)(h)
        
        h = Dense(512)(h)
        h = Activation(activation)(h)
        h = BatchNormalization(momentum=0.8)(h)
        
        h = Dense(1024)(h)
        h = Activation(activation)(h)
        
        outputs = Dense(self.x_dim[0], activation='tanh')(h)

        return Model([z, y], outputs, name='generator')

    def build_discriminator(self):
        activation = leakyrelu
        x = Input(shape = self.x_dim)
        y = Input(shape = self.y_dim)
        inputs = Concatenate()([x, y])
        
        h = Dense(512)(inputs)
        h = Activation(activation)(h)
        
        h = Dense(256)(h)
        h = Activation(activation)(h)
        
        outputs = Dense(1)(h)
        return Model([x, y], outputs, name='discriminator')
    
    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        
        fake_labels = tf.ones((batch_size, 1))
        real_labels = tf.ones((batch_size, 1))*0
        labels = tf.concat([real_labels, fake_labels], 0)
        
        noise = tf.random.normal((batch_size, self.z_dim[0]))
        
        # discriminator
        with tf.GradientTape() as tape:
            fake = self.generator([noise, y])
            all_x = tf.concat([x, fake], 0)
            all_y = tf.concat([y, y,], 0)
            preds = self.discriminator([all_x, all_y])

            d_loss = self.loss_fn(labels, preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optim.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # generator
        with tf.GradientTape() as tape:
            fake = self.generator([noise, y])
            preds = self.discriminator([fake, y])
            
            g_loss = self.loss_fn(real_labels, preds)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optim.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {'d_loss': d_loss, 'g_loss': g_loss}

