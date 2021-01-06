
import numpy as np
import os
import random
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input, Reshape, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Model, Sequential, load_model

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
tf.config.experimental.set_visible_devices([], 'GPU') # if cannot find gpu device


class GAN(keras.Model):
    def __init__(self, x_dim, z_dim): # dim: tuple
        super(GAN, self).__init__()
        self.x_dim  = x_dim
        self.z_dim = z_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def compile(self, g_optim, d_optim, loss_fn):
        super(GAN, self).compile()
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.loss_fn = loss_fn
        
    def build_generator(self): 
        activation = 'relu'
        inputs = Input(shape=self.z_dim)
        x = Dense(256)(inputs)
        x = Activation(activation)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(512)(x)
        x = Activation(activation)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(1024)(x)
        x = Activation(activation)(x)
        
        outputs = Dense(self.x_dim[0], activation='tanh')(x)
        
        return Model(inputs, outputs, name='generator')

    def build_discriminator(self):
        activation = leakyrelu
        inputs = Input(shape = self.x_dim)
        x = Dense(512)(inputs)
        x = Activation(activation)(x)
        x = Dense(256)(x)
        x = Activation(activation)(x)
        
        outputs = Dense(1)(x)

        return Model(inputs, outputs, name='discriminator')
    
    def train_step(self, x):
        batch_size = tf.shape(x)[0]
        
        fake_labels = tf.ones((batch_size, 1))
        real_labels = tf.ones((batch_size, 1))*0
        labels = tf.concat([real_labels, fake_labels], 0)
        
        noise = tf.random.normal((batch_size, self.z_dim[0]))
        
        # discriminator
        with tf.GradientTape() as tape:
            fake = self.generator(noise)
            all_x = tf.concat([x, fake], 0)
            preds = self.discriminator(all_x)
            
            d_loss = self.loss_fn(labels, preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optim.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # generator
        with tf.GradientTape() as tape:
            fake = self.generator(noise)
            preds = self.discriminator(fake)

            g_loss = self.loss_fn(real_labels, preds)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optim.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {'d_loss': d_loss, 'g_loss': g_loss}